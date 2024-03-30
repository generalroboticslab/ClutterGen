import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Normal
from torch_utils import quat_from_euler_xyz
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
    def __init__(self, channels=3, output_dim=1024):
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
            nn.ReLU(),
            layer_init(nn.Linear(256, output_dim)), 
            nn.LayerNorm(output_dim), 
            nn.ReLU()
        )

    def forward(self, x):
        # x: B, N, 3
        x = self.mlp(x) # B, N, 256
        x = torch.max(x, 1)[0] # B, 256
        return x


class Base_Actor(nn.Module):
    def __init__(self, mlp_input, num_action_logits, dropout=0.) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(mlp_input, 256)),
            nn.LayerNorm(256), 
            nn.ReLU(),
            nn.Dropout(dropout),
            layer_init(nn.Linear(256, 128)),
            nn.LayerNorm(128), 
            nn.ReLU(),
            nn.Dropout(dropout),
            layer_init(nn.Linear(128, 64)),
            nn.LayerNorm(64), 
            nn.ReLU(),
            nn.Dropout(dropout),
            layer_init(nn.Linear(64, num_action_logits))
        )
    
    def forward(self, x):
        return self.actor(x)


class StablePlacementPolicy_Determ(nn.Module):
    def __init__(self, mlp_input=2048, action_dim=4, device=None) -> None:
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # PointNet
        assert mlp_input % 2 == 0, "mlp_input must be even"
        self.pc_ft_dim = mlp_input // 2
        self.scene_pc_encoder = PC_Encoder(output_dim=self.pc_ft_dim)
        self.qr_obj_pc_encoder = PC_Encoder(output_dim=self.pc_ft_dim)
        # Linear
        self.num_action_logits = action_dim
        self.actor = Base_Actor(mlp_input, self.num_action_logits)

    def forward(self, scene_pc, qr_obj_pc):
        # PC-Net Points xyz: input points position data, [B, C, N]
        # While our pc input is [B, N, C] so we need transpose (if we use pointNet)
        # scene_pc = scene_pc.transpose(1, 2)
        # qr_obj_pc = qr_obj_pc.transpose(1, 2)
        scene_pc_feature = self.scene_pc_encoder(scene_pc)
        qr_obj_pc_feature = self.qr_obj_pc_encoder(qr_obj_pc)
        feature = torch.cat([scene_pc_feature, qr_obj_pc_feature], dim=1)
        action = self.actor(feature)
        pred_pos, pred_rotz = action[:, :3], action[:, 3]
        raw_pred_quat = quat_from_euler_xyz(torch.zeros_like(pred_rotz), torch.zeros_like(pred_rotz), pred_rotz)
        # Create a new tensor for the normalized quaternion, avoiding in-place modification
        pred_quat = F.normalize(raw_pred_quat, p=2, dim=1)
        pred_pose = torch.cat([pred_pos, pred_quat], dim=1)
        return pred_pose, None


class StablePlacementPolicy_Beta(nn.Module):
    def __init__(self, mlp_input=2048, action_dim=4, device=None) -> None:
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # PointNet
        assert mlp_input % 2 == 0, "mlp_input must be even"
        self.pc_ft_dim = mlp_input // 2
        self.scene_pc_encoder = PC_Encoder()
        self.qr_obj_pc_encoder = PC_Encoder()
        # Linear
        self.num_action_logits = action_dim * 2 # * 2 for alpha and beta
        self.actor = Base_Actor(mlp_input, self.num_action_logits)
    
    def forward(self, scene_pc, qr_obj_pc):
        # PC-Net Points xyz: input points position data, [B, C, N]
        # While our pc input is [B, N, C] so we need transpose
        scene_pc_feature = self.scene_pc_encoder(scene_pc)
        qr_obj_pc_feature = self.qr_obj_pc_encoder(qr_obj_pc)
        feature = torch.cat([scene_pc_feature, qr_obj_pc_feature], dim=1)
        action_logalpha_logbeta = self.actor(feature)
        action_logalpha, action_logbeta = torch.chunk(action_logalpha_logbeta, 2, dim=-1)
        action_alpha = torch.exp(action_logalpha)
        action_beta = torch.exp(action_logbeta)
        probs = Beta(action_alpha, action_beta)
        action = probs.rsample()
        pred_pos, pred_rotz = action[:, :3], action[:, 3]
        raw_pred_quat = quat_from_euler_xyz(torch.zeros_like(pred_rotz), torch.zeros_like(pred_rotz), pred_rotz)
        # Create a new tensor for the normalized quaternion, avoiding in-place modification
        pred_quat = F.normalize(raw_pred_quat, p=2, dim=1)
        pred_pose = torch.cat([pred_pos, pred_quat], dim=1)
        return pred_pose, probs.entropy().sum(dim=1)
    

class StablePlacementPolicy_Normal(nn.Module):
    def __init__(self, mlp_input=2048, action_dim=4, device=None) -> None:
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # PointNet
        assert mlp_input % 2 == 0, "mlp_input must be even"
        self.pc_ft_dim = mlp_input // 2
        self.scene_pc_encoder = PC_Encoder()
        self.qr_obj_pc_encoder = PC_Encoder()
        # Linear
        self.num_action_logits = action_dim # for mean
        self.actor = Base_Actor(mlp_input, self.num_action_logits)

        self.logstd = nn.Parameter(torch.zeros(action_dim), requires_grad=True)
    
    def forward(self, scene_pc, qr_obj_pc):
        # PC-Net Points xyz: input points position data, [B, C, N]
        # While our pc input is [B, N, C] so we need transpose
        scene_pc_feature = self.scene_pc_encoder(scene_pc)
        qr_obj_pc_feature = self.qr_obj_pc_encoder(qr_obj_pc)
        feature = torch.cat([scene_pc_feature, qr_obj_pc_feature], dim=1)
        action_mean = self.actor(feature)
        action_std = torch.exp(self.logstd).expand_as(action_mean)
        probs = Normal(action_mean, action_std)
        action = probs.rsample()
        pred_pos, pred_rotz = action[:, :3], action[:, 3]
        raw_pred_quat = quat_from_euler_xyz(torch.zeros_like(pred_rotz), torch.zeros_like(pred_rotz), pred_rotz)
        # Create a new tensor for the normalized quaternion, avoiding in-place modification
        pred_quat = F.normalize(raw_pred_quat, p=2, dim=1)
        pred_pose = torch.cat([pred_pos, pred_quat], dim=1)
        return pred_pose, probs.entropy().sum(dim=1)
