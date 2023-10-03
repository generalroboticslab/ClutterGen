import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
import timm
from torchvision import transforms

MIN_STD = 0.05
INITIAL_STD = 1


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class LSTM_Linear(nn.Module):
    def __init__(self, input_size, hidden_size, num_lstm_layers, num_linear_layers, output_size, batch_first=True, init_std=0.01):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_hidden_layers=num_lstm_layers, batch_first=batch_first)
        self.linear = nn.Sequential()
        for _ in range(num_linear_layers-1):
            self.linear.append(layer_init(nn.Linear(hidden_size, hidden_size)))
            self.linear.append(nn.Tanh())
        self.linear.append(layer_init(nn.Linear(hidden_size, output_size), std=init_std))

    def forward(self, x):
        sequence_features, (hn, cn) = self.lstm(x) # hn -> hidden state; cn -> cell state
        output = self.linear(sequence_features[:, -1, :]) # only use the last layer output
        return output
    

class Transfromer_Linear(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_len, num_transf_layers, num_linear_layers, output_size, batch_first=True, init_std=0.01) -> None:
        super().__init__()
        self.transformer = nn.TransformerEncoderLayer(d_model=input_size, nhead=1, dim_feedforward=hidden_size, batch_first=batch_first)
        self.lienar = nn.Sequential()
        for i in range(num_linear_layers-1):
            input_shape = sequence_len * input_size if i==0 else hidden_size
            self.lienar.append(layer_init(nn.Linear(input_shape, hidden_size)))
            self.lienar.append(nn.Tanh())
        self.lienar.append(layer_init(nn.Linear(hidden_size, output_size), std=init_std))
    
    def forward(self, x):
        embeddings = self.transformer(x)
        embeddings = torch.flatten(embeddings, start_dim=1)
        output = self.lienar(embeddings)
        return output
    

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, output_size, init_std=0.01):
        super().__init__()
        self.linear = nn.Sequential()
        for i in range(num_hidden_layers):
            input_shape = input_size if i==0 else hidden_size
            self.linear.append(layer_init(nn.Linear(input_shape, hidden_size)))
            self.linear.append(nn.Tanh())
        self.linear.append(layer_init(nn.Linear(hidden_size, output_size), std=init_std))

    def forward(self, x):
        return self.linear(x)
    

class Vit_backbone(nn.Module):
    def __init__(self, in_chans=4, pretrained=True, tensor_in=True) -> None:
        super().__init__()
        self.vit = timm.create_model(
                        'vit_small_patch16_224', 
                        pretrained=pretrained,
                        num_classes=0,
                        in_chans=in_chans)   # remove classifier nn.Linear
        self.data_config = timm.data.resolve_model_data_config(self.vit)
        self.transforms = transforms.Compose([
            transforms.Resize(self.data_config['input_size'][-2:]),
            transforms.CenterCrop(self.data_config['input_size'][-2:]),
            transforms.Normalize(mean=torch.tensor([0.5000, 0.5000, 0.5000, 0.]), 
                                 std=torch.tensor([0.5000, 0.5000, 0.5000, 1.]))
        ])

    def forward(self, x):
        return self.vit(self.transforms(x))


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.envs = envs
        self.time_seq_obs = envs.time_seq_obs
        self.hidden_size = envs.hidden_size
        self.num_hidden_layers = envs.num_hidden_layers
        self.action_nvec = [len(envs.action_range)] * envs.action_dim[1]
        self.action_logits_num = sum(self.action_nvec)

        self.vision_backbone = Vit_backbone(pretrained=True, tensor_in=True).to(self.device)
        self.vision_output_size = self.vision_backbone(torch.randn(1, *self.envs.single_img_obs_dim[1:]).to(self.device)).shape[-1]
        self.robot_backbone = MLP(self.envs.single_proprioception_dim[1], self.hidden_size, 1, self.hidden_size).to(self.device)
        self.agent_input_size = self.vision_output_size + self.hidden_size

        # Use MLP
        self.critic = MLP(
            input_size=self.agent_input_size, 
            hidden_size=self.hidden_size, 
            num_hidden_layers=self.num_hidden_layers, 
            output_size=1,
            init_std=0.01).to(self.device)
        
        self.actor = MLP(
            input_size=self.agent_input_size, 
            hidden_size=self.hidden_size, 
            num_hidden_layers=self.num_hidden_layers, 
            output_size=self.action_logits_num,
            init_std=1.0).to(self.device)
            

    def get_embedding(self, obs):
        vis_obs, robot_obs = obs['image'].to(self.device), obs['proprioception'].to(self.device)
        print(vis_obs[0, -1, :, :])
        vis_emb = self.vision_backbone(vis_obs)
        robot_emb = self.robot_backbone(robot_obs)
        return torch.cat([vis_emb, robot_emb], dim=-1)


    def get_value(self, x):
        return self.critic(x)


    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        split_logits = torch.split(logits, self.action_nvec, dim=1)
        multi_categoricals = [Categorical(logits=logits) for logits in split_logits]
        if action is None:
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        return action.T, logprob.sum(0), entropy.sum(0), self.critic(x) # why transpose: torch.stack makes space as [actions, env] but we need [env, actions]


    def select_action(self, state):
        state = state.to(self.device)  # size([sequence_len, state_dim]) --> size([1, sequence_len, state_dim])
        logits = self.actor(state)
        split_logits = torch.split(logits, self.action_nvec, dim=1)
        multi_categoricals = [Categorical(logits=logits) for logits in split_logits]
        action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        return action.T


    def set_mode(self, mode='train'):
        if mode == 'train': self.train()
        elif mode == 'eval': self.eval()
    

    def save_checkpoint(self, folder_path, folder_name, suffix="", ckpt_path=None):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if ckpt_path is None:
            ckpt_path = "{}/{}_{}".format(folder_path, folder_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save(self.state_dict(), ckpt_path)


    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False, load_memory=True, map_location='cuda:0'):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location=map_location)
            self.load_state_dict(checkpoint)

            if evaluate: self.set_mode('eval')
            else: self.set_mode('train')


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


if __name__ == "__main__":

    class TestModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor_std = nn.Parameter(torch.zeros(1, 2), requires_grad=False)
