import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# PointNet
from PointNet_Model.pointnet2_cls_ssg import get_model


def layer_init(layer, std=math.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AutoFlatten(nn.Module):
    def __init__(self, start_dim=1):
        super().__init__()
        self.start_dim = start_dim
    
    def forward(self, x):
        return torch.flatten(x, start_dim=self.start_dim)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, use_relu=True, output_layernorm=False, init_std=1., auto_flatten=False, flatten_start_dim=1):
        super().__init__()
        self.activation = nn.ReLU() if use_relu else nn.Tanh()
        self.mlp = nn.Sequential() \
                   if not auto_flatten \
                   else nn.Sequential(AutoFlatten(start_dim=flatten_start_dim))
        for i in range(num_layers-1):
            input_shape = input_size if i==0 else hidden_size
            self.mlp.append(layer_init(nn.Linear(input_shape, hidden_size)))
            self.mlp.append(nn.LayerNorm(hidden_size))
            self.mlp.append(self.activation)
        self.mlp.append(layer_init(nn.Linear(hidden_size, output_size), std=init_std))
        if output_layernorm:
            self.mlp.append(nn.LayerNorm(output_size))
    
    def forward(self, x):
        return self.mlp(x)


class PC_Encoder(nn.Module):
    def __init__(self, channels=3, output_dim=64):
        super().__init__()
        # We only use xyz (channels=3) in this work
        # while our encoder also works for xyzrgb (channels=6) in our experiments
        self.mlp = nn.Sequential(
            layer_init(nn.Linear(channels, 64)), 
            nn.LayerNorm(64), 
            nn.ReLU(),
            layer_init(nn.Linear(64, 128)), 
            nn.LayerNorm(128), 
            nn.ReLU(),
            layer_init(nn.Linear(128, 256)), 
            nn.LayerNorm(256), 
            nn.ReLU()
        )
        self.projection = nn.Sequential(
            layer_init(nn.Linear(256, output_dim)), 
            nn.LayerNorm(output_dim)
        )
    def forward(self, x):
        # x: B, N, 3
        x = self.mlp(x) # B, N, 256
        x = torch.max(x, 1)[0] # B, 256
        x = self.projection(x) # B, Output_dim
        return x


class StablePlacementModel(nn.Module):
    def __init__(self, num_linear=3, mlp_input=2048, mlp_output=7, device=None) -> None:
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # PointNet
        self.scene_pc_encoder = get_model(num_class=40, normal_channel=False) # num_classes is used for loading checkpoint to make sure the model is the same
        self.scene_pc_encoder.load_checkpoint(ckpt_path="../PointNet_Model/checkpoints/best_model.pth", evaluate=False, map_location=self.device)
        self.qr_obj_pc_encoder = get_model(num_class=40, normal_channel=False)
        self.qr_obj_pc_encoder.load_checkpoint(ckpt_path="../PointNet_Model/checkpoints/best_model.pth", evaluate=False, map_location=self.device)
        # Linear
        self.mlp = nn.Sequential(
            layer_init(nn.Linear(mlp_input, 1024)),
            nn.LayerNorm(1024), 
            nn.ReLU(),
            layer_init(nn.Linear(1024, 512)),
            nn.LayerNorm(512), 
            nn.ReLU(),
            layer_init(nn.Linear(512, mlp_output))
        )

    
    def forward(self, scene_pc, qr_obj_pc):
        # PC-Net Points xyz: input points position data, [B, C, N]
        # While our pc input is [B, N, C] so we need transpose
        scene_pc = scene_pc.transpose(1, 2)
        qr_obj_pc = qr_obj_pc.transpose(1, 2)
        scene_pc_feature = self.scene_pc_encoder(scene_pc)
        qr_obj_pc_feature = self.qr_obj_pc_encoder(qr_obj_pc)
        feature = torch.cat([scene_pc_feature, qr_obj_pc_feature], dim=1)
        pred_pose = self.mlp(feature)
        # Create a new tensor for the normalized quaternion, avoiding in-place modification
        normalized_quaternion = F.normalize(pred_pose[:, 3:], p=2, dim=1)
        # Concatenate the unchanged part of pred_pose with the normalized quaternion
        pred_pose = torch.cat([pred_pose[:, :3], normalized_quaternion], dim=1)
        return pred_pose

