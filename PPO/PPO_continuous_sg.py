import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

# PointNet
from PointNet_Model.pointnet2_cls_ssg import get_model

import copy
import math

MIN_STD = 0.05
INITIAL_STD = 1


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class LSTM_Linear(nn.Module):
    def __init__(self, input_size, hidden_size, num_lstm_layers, num_linear_layers, output_size, batch_first=True, init_std=0.01):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_lstm_layers, batch_first=batch_first)
        self.linear = MLP(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_linear_layers,
            init_std=init_std
        )

    def forward(self, x):
        sequence_features, (hn, cn) = self.lstm(x) # hn -> hidden state; cn -> cell state
        output = self.linear(sequence_features[:, -1, :]) # only use the last layer output
        return output
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, batch_first=True):
        super().__init__()
        # Pad d_model to be even if it's odd
        self.batch_first = batch_first
        self.pad = (d_model % 2 != 0)
        if self.pad:
            d_model += 1

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        if not batch_first: # (batch_size, seq_len, d_model) -> (seq_len, batch_size, d_model)
            pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.pad:
            # Pad the input with zeros on the last dimension to make it even
            zero_padding = torch.zeros((x.size(0), x.size(1), 1), device=x.device)
            x = torch.cat((x, zero_padding), dim=-1)

        if self.batch_first: x = x + self.pe[:, :x.size(1)]
        else: x = x + self.pe[:x.size(0), :]

        if self.pad:
            # Remove the padded 0 after adding the positional encoding
            x = x[..., :-1]
        return x


class Transfromer_Linear(nn.Module):
    def __init__(self, input_size, hidden_size, num_transf_layers, num_linear_layers, output_size, nhead=1, batch_first=True, init_std=0.01) -> None:
        super().__init__()
        self.positional_encoding = PositionalEncoding(input_size, batch_first=batch_first)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_size, 
            nhead=nhead, 
            dim_feedforward=hidden_size, 
            batch_first=batch_first
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer, 
            num_layers=num_transf_layers, 
            norm=nn.LayerNorm(input_size)
        )
        self.linear = MLP(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_linear_layers,
            init_std=init_std
        )
    
    def forward(self, x):
        x = self.positional_encoding(x)
        embeddings = self.transformer(x)
        last_embedding = embeddings[:, -1, :] # only use the last layer output
        output = self.linear(last_embedding)
        return output
    

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, use_relu=False, init_std=1., auto_flatten=False, flatten_start_dim=1):
        super().__init__()
        self.activation = nn.ReLU() if use_relu else nn.Tanh()
        self.mlp = nn.Sequential() \
                   if not auto_flatten \
                   else nn.Sequential(AutoFlatten(start_dim=flatten_start_dim))
        for i in range(num_layers-1):
            input_shape = input_size if i==0 else hidden_size
            self.mlp.append(layer_init(nn.Linear(input_shape, hidden_size)))
            self.mlp.append(self.activation)
        self.mlp.append(layer_init(nn.Linear(hidden_size, output_size), std=init_std))
    
    def forward(self, x):
        return self.mlp(x)
    

class AutoFlatten(nn.Module):
    def __init__(self, start_dim=1):
        super().__init__()
        self.start_dim = start_dim
    
    def forward(self, x):
        return torch.flatten(x, start_dim=self.start_dim)


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.envs = envs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activation = nn.Tanh() if not envs.args.use_relu else nn.ReLU()
        self.deterministic = envs.args.deterministic
        self.hidden_size = envs.args.hidden_size
        self.action_logits_num = envs.action_shape[1] # 2 for mean and std

        # Layer Number
        self.traj_hist_encoder_num_linear = 2; self.traj_hist_encoder_num_transf = 2
        self.seq_obs_encoder_num_linear = 2; self.seq_obs_encoder_num_transf = 2
        self.critic_num_linear = 5; self.actor_num_linear = 5
        
        # PointNet
        self.pc_batchsize = envs.args.pc_batchsize
        self.pc_extractor = get_model(num_class=40, normal_channel=False).to(self.device) # num_classes is used for loading checkpoint to make sure the model is the same
        self.pc_extractor.load_checkpoint(ckpt_path="PointNet_Model/checkpoints/best_model.pth", evaluate=True, map_location=self.device)

        # Trajectory history encoder and Large sequence observation encoder
        if envs.args.use_tf_traj_encoder:
            self.traj_hist_encoder = Transfromer_Linear(
                input_size=envs.traj_history_shape[1], 
                hidden_size=self.hidden_size, 
                num_transf_layers=self.traj_hist_encoder_num_transf,
                num_linear_layers=self.traj_hist_encoder_num_linear, 
                output_size=envs.history_ft_dim, 
                batch_first=True, 
                init_std=1.0
            ).to(self.device)
        else:
            self.traj_hist_encoder = MLP(
                input_size=np.prod(envs.traj_history_shape), 
                hidden_size=self.hidden_size, 
                output_size=envs.history_ft_dim, 
                num_layers=self.traj_hist_encoder_num_linear, 
                use_relu=envs.args.use_relu,
                init_std=1.0, auto_flatten=True
            ).to(self.device)
        
        if envs.args.use_tf_seq_obs_encoder:
            self.seq_obs_encoder = Transfromer_Linear(
                input_size=envs.post_act_hist_qr_ft_shape[2], 
                hidden_size=self.hidden_size, 
                num_transf_layers=self.seq_obs_encoder_num_transf,
                num_linear_layers=self.seq_obs_encoder_num_linear, 
                output_size=envs.seq_info_ft_dim, 
                nhead=8,
                batch_first=True, 
                init_std=1.0
            ).to(self.device)
        else:
            self.seq_obs_encoder = MLP(
                input_size=np.prod(envs.post_act_hist_qr_ft_shape[1:]), 
                hidden_size=self.hidden_size, 
                output_size=envs.seq_info_ft_dim, 
                num_layers=self.seq_obs_encoder_num_linear, 
                use_relu=envs.args.use_relu,
                init_std=1.0, auto_flatten=True
            ).to(self.device)
        
        # Use MLP for the critic and actor
        self.critic = MLP(
            input_size=envs.post_observation_shape[1],
            hidden_size=self.hidden_size,
            output_size=1,
            num_layers=self.critic_num_linear,
            use_relu=envs.args.use_relu,
            init_std=1.0,
        ).to(self.device)
        self.actor = MLP(
            input_size=envs.post_observation_shape[1],
            hidden_size=self.hidden_size,
            output_size=self.action_logits_num,
            num_layers=self.actor_num_linear,
            use_relu=envs.args.use_relu,
            init_std=0.01,
        ).to(self.device)

        # is_leaf problem for nn.parameters/must set to() within it
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_shape)).to(self.device), requires_grad=True) 

        self.to(envs.tensor_dtype)

    
    def seq_obs_ft_extract(self, seq_obs):
        """
        Preprocess the sequence observation and extract the features
        input seq_obs: (Env, Seq, Dim1)
        output seq_obs_ft: (Env, Dim2)
        """
        if self.envs.args.use_traj_encoder:
            qr_region_hist = seq_obs[:, :, self.envs.qr_region_slice]
            act_hist = seq_obs[:, :, self.envs.action_slice]
            traj_history = seq_obs[:, :, self.envs.traj_history_slice].view(-1, *self.envs.traj_history_shape)
            traj_hist_ft = self.traj_hist_encoder(traj_history)
            traj_hist_ft = traj_hist_ft.view(*seq_obs.shape[:2], self.envs.history_ft_dim)
            seq_obs = torch.cat([qr_region_hist, act_hist, traj_hist_ft], dim=-1)
        
        if self.envs.args.use_seq_obs_encoder:
            seq_obs = self.seq_obs_encoder(seq_obs)
        else:
            # Flatten the sequence observation from (Env, Seq, Dim) to (Env, Seq*Dim)
            seq_obs = torch.flatten(seq_obs, start_dim=1)
        return seq_obs
    

    def preprocess_pc_update_tensor(self, all_envs_scene_ft_tensor, all_envs_obj_ft_tensor, infos, use_mask=False):
        if not self.envs.args.use_pc_extractor: return

        scene_pc_buf = []; scene_pc_update_env_ids = []
        for i, info in enumerate(infos):
            if use_mask:
                indicator = info["pc_change_indicator"]
                if not indicator: 
                    continue

            scene_pc, init_scene_pc_ft, obj_pc_ft = info["selected_qr_scene_pc"], info["selected_init_qr_scene_ft"], info["selected_obj_pc_ft"]
            # Update object point cloud features
            all_envs_obj_ft_tensor[i] = obj_pc_ft.to(self.device)
            # Update init scene point cloud features, which does not require re-extracted
            if init_scene_pc_ft is not None:
                all_envs_scene_ft_tensor[i] = init_scene_pc_ft.to(self.device)
                continue
            
            scene_pc_update_env_ids.append(i)
            # Make sure the scene point cloud has the same number of points (object already has the same number of points)
            if scene_pc.shape[0] < self.envs.args.max_num_scene_points:
                scene_pc = np.concatenate([scene_pc, np.zeros((self.envs.args.max_num_scene_points-scene_pc.shape[0], scene_pc.shape[1]))], axis=0)
            scene_pc_buf.append(np.expand_dims(scene_pc, axis=0))
        if len(scene_pc_update_env_ids) == 0: return
        print(f"Update Scene Point Cloud for {scene_pc_update_env_ids}")
        scene_pc_buf = np.concatenate(scene_pc_buf, axis=0)
        # PC-Net Points xyz: input points position data, [B, C, N]
        # While our pc input is [B, N, C] so we need transpose
        # Batch operation to increase the maximum environments
        # Update scene point cloud features which requires re-extracted
        for i in range(0, len(scene_pc_update_env_ids), self.pc_batchsize):
            with torch.no_grad():
                update_env_ids_minibatch = scene_pc_update_env_ids[i:i+self.pc_batchsize]
                scene_pc_minibatch = torch.Tensor(scene_pc_buf[i:i+self.pc_batchsize]).to(self.device).transpose(1, 2)
                scene_pc_ft = self.pc_extractor(scene_pc_minibatch)
                all_envs_scene_ft_tensor[update_env_ids_minibatch] = scene_pc_ft


    def get_value(self, obs_list):
        seq_obs, scene_ft_tensor, obj_ft_tensor = obs_list
        seq_obs_ft = self.seq_obs_ft_extract(seq_obs)
        x = torch.cat([seq_obs_ft, scene_ft_tensor, obj_ft_tensor], dim=-1)
        return self.critic(x)


    def get_action_and_value(self, obs_list, action=None):
        seq_obs, scene_ft_tensor, obj_ft_tensor = obs_list
        seq_obs_ft = self.seq_obs_ft_extract(seq_obs)
        # We currently use cat to combine all features; Later we can try to use attention to combine them
        x = torch.cat([seq_obs_ft, scene_ft_tensor, obj_ft_tensor], dim=-1)
        action_mean = self.actor(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = action_mean if self.deterministic else probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    

    def select_action(self, obs_list):
        seq_obs, scene_ft_tensor, obj_ft_tensor = obs_list
        seq_obs_ft = self.seq_obs_ft_extract(seq_obs)
        x = torch.cat([seq_obs_ft, scene_ft_tensor, obj_ft_tensor], dim=-1)
        action_mean = self.actor(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        action = action_mean if self.deterministic else probs.sample()
        return action, probs


    def set_mode(self, mode='train'):
        if mode == 'train': self.train()
        elif mode == 'eval': self.eval()
    

    def save_checkpoint(self, folder_path, folder_name, suffix="", ckpt_path=None):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if ckpt_path is None:
            ckpt_path = "{}/{}_{}".format(folder_path, folder_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        # Don't save the pc_extractor; we have weights offline
        filtered_state_dict = {k: v for k, v in self.state_dict().items() if 'pc_extractor' not in k}
        torch.save(filtered_state_dict, ckpt_path)


    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False, map_location='cuda:0'):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location=map_location)
            self.load_state_dict(checkpoint, strict=False)

            if evaluate: self.set_mode('eval')
            else: self.set_mode('train')


    # MultivariateNormal does not support bfloat16
    # def get_action_and_value(self, x, action=None):
    #     action_mean = self.actor(x)
    #     action_logstd = self.actor_logstd.expand_as(action_mean)
    #     action_std = torch.exp(action_logstd)
    #     cov_mat = torch.diag_embed(action_std)
    #     probs = MultivariateNormal(action_mean, cov_mat)
    #     if action is None:
    #         action = action_mean if self.deterministic else probs.sample()
    #     return action, probs.log_prob(action), probs.entropy(), self.critic(x)



# Misc Functions
def control_len(lst, length=100): # control a list length to be a certain number
    if len(lst) <= length: return lst
    else: return lst[len(lst)-length:]


def update_tensor_buffer(buffer, new_v):
    len_v = len(new_v)
    if len_v > len(buffer):
        buffer[:] = new_v[len_v-len(buffer):]
    else:
        buffer[:-len_v] = buffer[len_v:].clone()
        buffer[-len_v:] = new_v


def convert_time(relative_time):
    relative_time = int(relative_time)
    hours = relative_time // 3600
    left_time = relative_time % 3600
    minutes = left_time // 60
    seconds = left_time % 60
    return f'{hours}:{minutes}:{seconds}'


if __name__ == "__main__":
    class TestModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor_std = nn.Parameter(torch.zeros(1, 2), requires_grad=False)
