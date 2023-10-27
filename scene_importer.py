from isaacgym_utils import euler_to_transform_matrix, unravel_index
from isaacgym.torch_utils import *
from PIL import Image
import requests
import numpy as np
from torchvision import transforms
from PIL import Image
import torch
import open3d as o3d

# DINOSAM Import
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict
from huggingface_hub import hf_hub_download
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


class SceneImporter:
    def __init__(self) -> None:
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build Model
        # torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo
        repo = "isl-org/ZoeDepth"
        # Zoe_N; NY / Kitti / NY and Kitti
        self.depth_estimator = torch.hub.load(repo, "ZoeD_N", pretrained=True).to(self.device)
        self.image_preprocess = transforms.Compose([
                                    transforms.ToTensor()
                                    ])
        
        fx = 422.364  # Focal length in x-direction
        fy = 422.364  # Focal length in y-direction
        cx = 192.0   # X-coordinate of the principal point
        cy = 192.0   # Y-coordinate of the principal point

        # # Create the intrinsic matrix
        # intrinsic_matrix = 
        self.intrinsic_matrix = np.array([[[fx, 0, cx],
                                          [0, fy, cy],
                                          [0, 0, 1]]])
        
        # # Create the extrinsic matrix
        self.extrinsic_matrix = np.array([[[1, 0, 0, 0],
                                          [0, 1, 0, 0],
                                          [0, 0, 1, 0],
                                          [0, 0, 0, 1]]])
    

    def preprocess_image(self, images_list: list, tonumpy=False):
        images_list = [self.image_preprocess(image) for image in images_list]
        images= torch.stack(images_list, dim=0).to(self.device)
        if tonumpy: images = images.cpu().numpy()
        return images


    def get_depth(self, images: torch.Tensor):
        if len(images.shape) == 3: images = images.unsqueeze(0)
        with torch.no_grad():
            prediction = self.depth_estimator.infer(images)
        return prediction
    

    def get_pc_from_rgbd(self, rgb_images, depth_images, intrinsic_matrix, extrinsic_pose=None, masks=None, downsample=None):
        # Convert input NumPy arrays to PyTorch tensors
        depth_images_tensor = self.to_torch(depth_images, dtype=torch.float32)
        rgb_images_tensor = self.to_torch(rgb_images, dtype=torch.float32)
        intrinsic_matrix_tensor = self.to_torch(intrinsic_matrix, dtype=torch.float32)
        
        # Extract height and width from the RGB image; (batch, C, H, W)
        batch_size, _, height, width = rgb_images.shape
        
        # Create 2D grids representing pixel coordinates; v is height, u is width
        if downsample is not None and downsample < height * width:
            point_indices = torch.randint(0, height*width, (downsample,)).to(self.device)
            point_indexes = unravel_index(point_indices, (height, width))
            v, u = point_indexes[:, 0], point_indexes[:, 1]
            depth_images_tensor = depth_images_tensor[:, :, v, u]
            rgb_images_tensor = rgb_images_tensor[:, :, v, u].unsqueeze(dim=-1) # (batch, C, downsample, 1)
            masks = masks[:, v, u] if masks is not None else None
        else:
            v, u = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
            v, u = self.to_torch(v), self.to_torch(u)
        
        # Compute 3D coordinates
        x = ((u - intrinsic_matrix_tensor[:, 0, 2]) * depth_images_tensor / intrinsic_matrix_tensor[:, 0, 0])
        y = ((v - intrinsic_matrix_tensor[:, 1, 2]) * depth_images_tensor / intrinsic_matrix_tensor[:, 1, 1])
        z = depth_images_tensor
        
        # Stack the 3D coordinates and RGB values
        points_3d = torch.stack((x, y, z), dim=-1)
        colors = rgb_images_tensor
        
        # Reshape tensors to flatten them
        self.points_3d = points_3d.view(batch_size, -1, 3)
        self.colors = colors.permute(0, 2, 3, 1).view(batch_size, -1, 3) # colors should be (batch, H, W, C)
        self.masks = masks.view(batch_size, -1) if masks is not None else None

        if extrinsic_pose is not None:
            extrinsic_quat, extrinsic_trans = extrinsic_pose
            extrinsic_quat, extrinsic_trans = self.to_torch(extrinsic_quat), self.to_torch(extrinsic_trans)
            # extrinsic_matrix_tensor = extrinsic_matrix_tensor.repeat(batch_size, 1, 1) # (batch, 4, 4)
            # TODO: Dont compute inverse every time
            # inv_extrinsic_quat, inv_extrinsic_trans = tf_inverse(extrinsic_quat, extrinsic_trans)
            inv_extrinsic_quat, inv_extrinsic_trans = extrinsic_quat, extrinsic_trans
            # v_extrinsic_matrix_tensor = extrinsic_matrix_tensor
            self.points_3d = tf_apply(inv_extrinsic_quat, inv_extrinsic_trans, self.points_3d)
        
        return self.points_3d, self.colors, self.masks
    

    def get_pc_from_rgb(self, rgb_images: list, intrinsic_matrix, extrinsic_matrix=None, masks=None, negative_depth=False, downsample=None):
        processed_rgb_images = self.preprocess_image(rgb_images)
        depth_image = self.get_depth(processed_rgb_images)
        if negative_depth: depth_image = -depth_image
        return self.get_pc_from_rgbd(processed_rgb_images, depth_image, intrinsic_matrix, extrinsic_matrix, masks, downsample)

    
    def visualize_pc(self, index=0):
        # # Create an Open3D point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(self.points_3d[index, :, :].cpu().numpy())
        point_cloud.colors = o3d.utility.Vector3dVector(self.colors[index, :, :].cpu().numpy())

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1., origin=[0, 0, 0])

        # Visualize the point cloud using Open3D
        o3d.visualization.draw_geometries([point_cloud, coord_frame])

    
    def visualize_depth(self, depth_image: torch.Tensor):
        depth_img = depth_image.squeeze().cpu().numpy()
        formatted = (depth_img * 255 / np.max(depth_img)).astype('uint8')
        # Convert the PIL Image back to a NumPy array (if needed)
        # Display the depth image using matplotlib
        plt.imshow(formatted, cmap='gray')  # Assuming a grayscale depth image
        plt.colorbar()  # Add a color bar to show the depth values
        plt.axis('off')  # Turn off axis numbers and ticks
        plt.show()


    def get_obj_pos_bbox(self, points_3d, mask):
        obj_bbox_store, obj_pos_store = [], []
        no_ground_mask = mask[mask!=0]
        obj_class_indexes = torch.arange(no_ground_mask.min(), no_ground_mask.max()+1, device=self.device)
        for obj_index in obj_class_indexes:
            obj_mask = mask == obj_index
            obj_points = points_3d[obj_mask, :]
            if len(obj_points) == 0: 
                obj_bbox = torch.zeros((2, 3), device=self.device)
                obj_pos = torch.zeros((3,), device=self.device)
            else:
                obj_bbox = torch.stack([obj_points.min(dim=0)[0], obj_points.max(dim=0)[0]], dim=0)
                obj_pos = obj_bbox.mean(dim=0)

            obj_bbox_store.append(obj_bbox)
            obj_pos_store.append(obj_pos)
        return torch.stack(obj_bbox_store, dim=0), torch.stack(obj_pos_store, dim=0)
    

    def get_obj_pos_bbox_batch(self, points_3ds, masks):
        obj_bbox_store_batch, obj_pos_store_batch = [], []
        for batch_index in range(points_3ds.shape[0]):
            obj_bbox, obj_pos = self.get_obj_pos_bbox(points_3ds[batch_index], masks[batch_index])
            obj_bbox_store_batch.append(obj_bbox)
            obj_pos_store_batch.append(obj_pos)
        return torch.stack(obj_bbox_store_batch, dim=0), torch.stack(obj_pos_store_batch, dim=0)


    def to_torch(self, x, dtype=torch.float32):
        return torch.tensor(x, dtype=dtype, device=self.device, requires_grad=False)
    

class DinoSAM:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dino_ckpt_repo_id = "ShilongLiu/GroundingDINO"
        self.dino_ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
        self.dino_ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

        self.groundingdino_model = self.load_model_hf(self.dino_ckpt_repo_id, 
                                                      self.dino_ckpt_filenmae, 
                                                      self.dino_ckpt_config_filename, 
                                                      device=self.device)
        self.gdino_transform = transforms.Compose(
            [
                transforms.Resize([800], max_size=1333),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        
        self.sam_checkpoint = "assets/sem_checkpoints/sam_vit_b_01ec64.pth"
        self.sam_model_type = "vit_b"
        self.sam_predictor = SamPredictor(sam_model_registry[self.sam_model_type](checkpoint=self.sam_checkpoint).to(device=self.device))
        
        self.detected_catorgories = \
            """
            headset, oranges,
            mouse, cabinet, ipad, shoes, box, table, person, bicycle, 
            car , motorcycle , airplane , bus , train , truck , boat , 
            traffic light , fire hydrant , stop sign , parking meter , 
            bench , bird , cat , dog , horse , sheep , cow , elephant , 
            bear , zebra , giraffe , backpack , umbrella , handbag , tie , 
            suitcase , frisbee , skis , snowboard , sports ball , kite , 
            baseball bat , baseball glove , skateboard , surfboard , 
            tennis racket , bottle , wine glass , cup , fork , knife , 
            spoon , bowl , banana , apple , sandwich , orange , 
            broccoli , carrot , hot dog , pizza , donut , cake , 
            chair , couch , potted plant , bed , dining table , 
            toilet , tv , laptop , mouse , remote , keyboard , 
            cell phone , microwave , oven , toaster , sink , 
            refrigerator , book , clock , vase , scissors , 
            teddy bear , hair drier , toothbrush"""


    def load_model_hf(self, repo_id, filename, ckpt_config_filename, device='cpu'):
        cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

        args = SLConfig.fromfile(cache_config_file) 
        model = build_model(args)
        args.device = device

        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location='cpu')
        log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        print("Model loaded from {} \n => {}".format(cache_file, log))
        _ = model.eval()
        return model
    

    def dino_detect(self, image, text_prompt=None, box_threshold = 0.3, text_threshold = 0.25):
        text_prompt = text_prompt if text_prompt is not None else self.detected_catorgories
        boxes, logits, phrases = predict(
            model=self.groundingdino_model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )
        return boxes, logits, phrases
    
    
    def dino_anotate(self, image, boxes, logits, phrases):
        annotated_frame = annotate(image_source=image, boxes=boxes, logits=logits, phrases=phrases)
        annotated_frame = annotated_frame[...,::-1] # BGR to RGB
        return annotated_frame
    

    def sam_segment(self, image, boxes, squeeze=True):
        image = np.asarray(image)
        self.sam_predictor.set_image(image)
        H, W, _ = image.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy.to(self.device), image.shape[:2])
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
            )
        
        # ground is 0, other labels are from 1 to masks.size(0)+1
        index_tensor = torch.arange(1, masks.size(0)+1).reshape(-1, 1, 1, 1).to(self.device)
        index_masks = masks * index_tensor

        if squeeze:
            index_masks = index_masks.max(dim=0)[0]
        return masks, index_masks
    

    def draw_sam_masks(self, masks, image, random_color=True):
        if masks.device != torch.device("cpu"): masks = masks.cpu()

        for i in range(masks.shape[0]):
            mask = masks[i, ...]
            if random_color:
                color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
            else:
                color = np.array([30/255, 144/255, 255/255, 0.6])
            
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

            annotated_frame_pil = Image.fromarray(image).convert("RGBA")
            mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

        return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))
    

    def get_masks_labels_from_image(self, image: Image.Image):
        image_tensor = self.gdino_transform(image)
        boxes, _, phrases = self.dino_detect(image_tensor)
        if len(phrases)>0:
            masks, index_masks = self.sam_segment(image, boxes)
        else:
            masks = torch.zeros(*image_tensor.shape, device=self.device).unsqueeze(dim=0)
            index_masks = torch.zeros(*image_tensor.shape, device=self.device)
        return masks, index_masks, phrases
    

def load_image_pil(image_path, resize=((384, 384))):
    image_source = Image.open(image_path).convert("RGB")
    image_source = image_source.resize(resize, Image.BILINEAR)
    return image_source


if __name__=="__main__":
    from matplotlib import pyplot as plt

    image_source = Image.open("assets/image_dataset/scratch/test4.jpg").convert("RGB").resize((256, 256), Image.BILINEAR)

    scene_importer = SceneImporter()
    # org_image = scene_importer.preprocess_image([image_source])
    # depth_image = scene_importer.get_depth(org_image)

    # scene_importer.visualize_depth(org_image[0, ...].permute(1, 2, 0))
    # scene_importer.visualize_depth(depth_image)

    # Compute the transformation matrices
    quat_rot = quat_conjugate(quat_from_euler_xyz(*torch.tensor([np.pi/2, -np.pi/2., 0.])))
    extrinsic_pose = (quat_rot, torch.tensor([0.0, 0.0, 0.0]))

    dinosam = DinoSAM()
    image_tensor = dinosam.gdino_transform(image_source)
    boxes, _, phrases = dinosam.dino_detect(image_tensor)
    masks, index_masks = dinosam.sam_segment(image_source, boxes)

    pc_pos, pc_color, pc_masks = scene_importer.get_pc_from_rgb([image_source], 
                                                                intrinsic_matrix=scene_importer.intrinsic_matrix, 
                                                                extrinsic_matrix=extrinsic_pose,
                                                                masks=index_masks, 
                                                                downsample=100)
    obj_bboxes, obj_center_poses = scene_importer.get_obj_pos_bbox_batch(pc_pos, pc_masks)
    # scene_importer.visualize_pc()


