import argparse
import os
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert


# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

# segment anything
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt

import PIL
import requests
import torch
from io import BytesIO

from huggingface_hub import hf_hub_download


class GD_SAM:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_dir = "assets/gd_sam_cache"

        # Grounding DINO model; Use huggingface_hub to download the model
        gd_ckpt_repo_id = "ShilongLiu/GroundingDINO"
        gd_ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
        gd_ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        self.gd_model = self.load_gd_model(gd_ckpt_repo_id, gd_ckpt_filenmae, gd_ckpt_config_filename, self.cache_dir, device=self.device)

        # SAM model; Use URL to download the model; By default, the model is vit_b
        sam_ckpt_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        self.sam_model = self.load_sam_model(sam_ckpt_url, self.cache_dir, device=self.device)

    
    @staticmethod
    def load_gd_model(repo_id, filename, ckpt_config_filename, cache_dir, device='cpu'):
        cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename, cache_dir=cache_dir)

        args = SLConfig.fromfile(cache_config_file) 
        args.device = device
        model = build_model(args)

        cache_file = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
        checkpoint = torch.load(cache_file, map_location=device)
        log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        print("Model loaded from {} \n => {}".format(cache_file, log))
        model.eval()
        return model
    

    @staticmethod
    def load_sam_model(sam_url, cache_dir, device='cuda'):
        sam_checkpoint_path = os.path.join(cache_dir, "sam", os.path.basename(sam_url))
        if not os.path.exists(sam_checkpoint_path):
            os.makedirs(os.path.dirname(sam_checkpoint_path), exist_ok=True)
            torch.hub.download_url_to_file(sam_url, sam_checkpoint_path)
        model_type = "vit_b"
        return SamPredictor(sam_model_registry[model_type](checkpoint=sam_checkpoint_path).to(device=device))
    

    @staticmethod
    def load_image_from_path(local_image_path="assets/image_dataset/scratch/test4.jpg"):
        """
        img_array_org: Original image array (H, W, C),
        img_tensor_trans: Transformed image tensor (C, H, W)
        """
        img_array_org, img_tensor_trans = load_image(local_image_path)
        return img_array_org, img_tensor_trans
    

    @staticmethod
    def img2tensor(img_array_org):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_PIL = Image.fromarray(img_array_org).convert("RGB")
        img_tensor_trans, _ = transform(image_PIL, None)
        return img_array_org, img_tensor_trans
    

    @staticmethod
    def draw_mask(mask, img_array_org, random_color=True):
        # mask: (H, W, 1); bool type
        # image: (H, W, C); np.uint8 type
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) * 255

        annotated_frame_pil = Image.fromarray(img_array_org).convert("RGBA")
        mask_image_pil = Image.fromarray(mask_image.astype(np.uint8)).convert("RGBA")
        return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))


    def gd_detect(self, img_tensor_trans, text_prompt, box_threshold = 0.5, text_threshold = 0.5, annotated = False):
        # detect object using grounding DINO
        boxes, logits, phrases = predict(
            model=self.gd_model,
            image=img_tensor_trans,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        return boxes, logits, phrases


    def sam_detect(self, img_array_org, boxes, annotated = False):
        """
        Input:
            image: Original image array (H, W, C),
            boxes: Detected boxes (B, 4) where N is the number of boxes
        Output:
            masks: Predicted masks (B, 1, H, W)
            scores: Predicted scores (B, 1)
        if multi_mask_output is True, masks will be 3 masks and scores will be 3 scores instead of 1
        """
        self.sam_model.set_image(img_array_org) # All process and read embedding inside
        H, W, _ = img_array_org.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H]).to(boxes.device)

        transformed_boxes = self.sam_model.transform.apply_boxes_torch(boxes_xyxy.to(self.sam_model.device), img_array_org.shape[:2])
        masks, scores, _ = self.sam_model.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
            )

        return masks, scores


    def predict(self, img_tensor_trans, img_array_org, text_prompt, box_threshold=0.7, text_threshold=0.7, gd_annotated=False, sam_annotated=False):
        # Predict the bounding box
        # boxes: Detected boxes (B, 4) where N is the number of boxes
        # masks: Predicted masks (B, 1, H, W)
        boxes, logits, phrases = self.gd_detect(img_tensor_trans, text_prompt, box_threshold, text_threshold, annotated=gd_annotated)
        if len(phrases) > 0: # If there is no detected object, skip the mask prediction
            masks, _ = self.sam_detect(img_array_org, boxes, annotated=sam_annotated)
        else:
            masks = torch.zeros((1, 1, *img_array_org.shape[:2]), device=self.device)
        
        # Original mask is bool type
        converted_masks = masks.to(torch.uint8) * 255

        annotated_img_array_org = img_array_org.copy() if gd_annotated or sam_annotated else None
        if gd_annotated:
            annotated_img_array_org = annotate(image_source=annotated_img_array_org, boxes=boxes, logits=logits, phrases=phrases)
            annotated_img_array_org = annotated_img_array_org[...,::-1] # BGR to RGB

        if sam_annotated:
            annotated_img_array_org = self.draw_mask(masks.cpu().numpy(), annotated_img_array_org)

        return converted_masks, phrases, annotated_img_array_org
    

if __name__ == "__main__":
    gd_sam = GD_SAM()
    img_array_org, img_tensor_trans = gd_sam.load_image_from_path("FoundationPose/demo_data/custom_test/IMG_7107.jpg")
    text_prompt = "red cube." # Use "." for each category
    masks, sem_labels, annotated_img_array_org = gd_sam.predict(img_tensor_trans, img_array_org, text_prompt, gd_annotated=True, sam_annotated=True)
    print(sem_labels)
    plt.imshow(annotated_img_array_org)
    plt.axis('off')
    plt.show()
    plt.imshow(masks[0].squeeze(0).cpu().numpy())
    plt.axis('off')
    plt.show()
