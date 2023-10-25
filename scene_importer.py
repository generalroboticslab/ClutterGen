from isaacgym_utils import euler_to_transform_matrix, unravel_index
from isaacgym.torch_utils import *
from PIL import Image
import requests
import numpy as np
from torchvision import transforms
from PIL import Image
import torch
import open3d as o3d
import torch


class SceneImporter:
    def __init__(self) -> None:
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build Model
        # torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo
        repo = "isl-org/ZoeDepth"
        # Zoe_N; NY / Kitti / NY and Kitti
        self.depth_estimator = torch.hub.load(repo, "ZoeD_N", pretrained=True).to(self.device)
        self.image_preprocess = transforms.Compose([
                                    transforms.Resize((384, 384)),
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
    

    def preprocess_image(self, images_list: list):
        images_list = [self.image_preprocess(image) for image in images_list]
        images_tensor= torch.stack(images_list, dim=0).to(self.device)
        return images_tensor


    def get_depth(self, images: torch.Tensor):
        if len(images.shape) == 3: images = images.unsqueeze(0)
        with torch.no_grad():
            prediction = self.depth_estimator.infer(images)
        return prediction
    

    def get_pc_from_rgbd(self, rgb_images, depth_images, intrinsic_matrix, extrinsic_pose=None, downsample=None):
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
            print("Downsample:", v.shape, u.shape, depth_images_tensor.shape, rgb_images_tensor.shape)
            depth_images_tensor = depth_images_tensor[:, :, v, u]
            rgb_images_tensor = rgb_images_tensor[:, :, v, u].unsqueeze(dim=-1) # (batch, C, downsample, 1)
            
        else:
            v, u = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
            v, u = self.to_torch(v), self.to_torch(u)
        
        # Compute 3D coordinates
        print("Converting:", u.shape, v.shape, depth_images_tensor.shape, intrinsic_matrix_tensor.shape)
        x = ((u - intrinsic_matrix_tensor[:, 0, 2]) * depth_images_tensor / intrinsic_matrix_tensor[:, 0, 0])
        y = ((v - intrinsic_matrix_tensor[:, 1, 2]) * depth_images_tensor / intrinsic_matrix_tensor[:, 1, 1])
        z = depth_images_tensor
        
        # Stack the 3D coordinates and RGB values
        points_3d = torch.stack((x, y, z), dim=-1)
        colors = rgb_images_tensor
        
        # Reshape tensors to flatten them
        self.points_3d = points_3d.view(batch_size, -1, 3)
        self.colors = colors.permute(0, 2, 3, 1).view(batch_size, -1, 3) # colors should be (batch, H, W, C)

        if extrinsic_pose is not None:
            extrinsic_quat, extrinsic_trans = extrinsic_pose
            extrinsic_quat, extrinsic_trans = self.to_torch(extrinsic_quat), self.to_torch(extrinsic_trans)
            # extrinsic_matrix_tensor = extrinsic_matrix_tensor.repeat(batch_size, 1, 1) # (batch, 4, 4)
            # TODO: Dont compute inverse every time
            # inv_extrinsic_quat, inv_extrinsic_trans = tf_inverse(extrinsic_quat, extrinsic_trans)
            inv_extrinsic_quat, inv_extrinsic_trans = extrinsic_quat, extrinsic_trans
            # v_extrinsic_matrix_tensor = extrinsic_matrix_tensor
            self.points_3d = tf_apply(inv_extrinsic_quat, inv_extrinsic_trans, self.points_3d)
        
        return self.points_3d, self.colors
    

    def get_pc_from_rgb(self, rgb_images: list, intrinsic_matrix, extrinsic_matrix=None, negative_depth=False, downsample=None):
        processed_rgb_images = self.preprocess_image(rgb_images)
        depth_image = self.get_depth(processed_rgb_images)
        if negative_depth: depth_image = -depth_image
        return self.get_pc_from_rgbd(processed_rgb_images, depth_image, intrinsic_matrix, extrinsic_matrix, downsample)

    
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


    def to_torch(self, x, dtype=torch.float32):
        return torch.tensor(x, dtype=dtype, device=self.device, requires_grad=False)
    

if __name__=="__main__":
    from matplotlib import pyplot as plt
    scene_importer = SceneImporter()

    org_image = Image.open("assets/image_dataset/scratch/test4.jpg")
    org_image = scene_importer.preprocess_image([org_image])
    depth_image = scene_importer.get_depth(org_image)

    # scene_importer.visualize_depth(org_image[0, ...].permute(1, 2, 0))
    # scene_importer.visualize_depth(depth_image)

    batch_size = 1  # Example batch size

    # Compute the transformation matrices
    quat_rot = quat_conjugate(quat_from_euler_xyz(*torch.tensor([np.pi/2, -np.pi/2., 0.])))
    extrinsic_pose = (quat_rot, torch.tensor([0.0, 0.0, 0.0]))

    scene_importer.get_pc_from_rgbd(org_image, depth_image, scene_importer.intrinsic_matrix, extrinsic_pose, downsample=10000)
    scene_importer.visualize_pc()
