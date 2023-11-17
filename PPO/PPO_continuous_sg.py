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

import copy

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


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.envs = envs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_lstm = envs.use_lstm if hasattr(envs, 'use_lstm') else False
        self.use_transformer = envs.use_transformer if hasattr(envs, 'use_transformer') else False
        self.lstm_layers = 1; self.linear_layers = 4
        self.deterministic = envs.args.deterministic
        self.hidden_size = envs.args.hidden_size

        self.action_logits_num = envs.action_shape[1]
        if self.use_lstm:  # Use LSTM and 1 MLP
            self.critic = LSTM_Linear(input_size=envs.observation_shape[2], hidden_size=self.hidden_size, num_lstm_layers=self.lstm_layers, 
                                      num_linear_layers=self.linear_layers, output_size=1, batch_first=True, init_std=1.0).to(self.device)
            self.actor = LSTM_Linear(input_size=envs.observation_shape[2], hidden_size=self.hidden_size, num_lstm_layers=self.lstm_layers, 
                                     num_linear_layers=self.linear_layers, output_size=self.action_logits_num, batch_first=True, init_std=0.01).to(self.device)

        elif self.use_transformer:
            self.critic = Transfromer_Linear(input_size=envs.observation_shape[2], hidden_size=self.hidden_size, sequence_len=envs.observation_shape[1], num_transf_layers=1,
                                             num_linear_layers=self.linear_layers, output_size=1, batch_first=True, init_std=1.0).to(self.device)
            self.actor = Transfromer_Linear(input_size=envs.observation_shape[2], hidden_size=self.hidden_size, sequence_len=envs.observation_shape[1], num_transf_layers=1,
                                             num_linear_layers=self.linear_layers, output_size=self.action_logits_num, batch_first=True, init_std=0.01).to(self.device)
        
        else:
            # Use MLP
            self.critic = nn.Sequential()
            for i in range(self.linear_layers):
                input_size = np.prod(envs.observation_shape[1:]) if i==0 else self.hidden_size
                self.critic.append(layer_init(nn.Linear(input_size, self.hidden_size)))
                self.critic.append(nn.Tanh())
            self.critic.append(layer_init(nn.Linear(self.hidden_size, 1), std=1.0))
            self.critic = self.critic.to(self.device)

            self.actor = nn.Sequential()
            for i in range(self.linear_layers):
                input_size = np.prod(envs.observation_shape[1:]) if i==0 else self.hidden_size
                self.actor.append(layer_init(nn.Linear(input_size, self.hidden_size)))
                self.actor.append(nn.Tanh())
            self.actor.append(layer_init(nn.Linear(self.hidden_size, self.action_logits_num), std=0.01))
            self.actor = self.actor.to(self.device)

        # Not train standard deviation, but only use linear schedular
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_shape)).to(self.device), requires_grad=True) # is_leaf problem for nn.parameters/must set to() within it

        self.to(self.envs.obs_dtype)


    def get_value(self, x):
        return self.critic(x)
    
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


    def get_action_and_value(self, x, action=None):
        action_mean = self.actor(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = action_mean if self.deterministic else probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    

    def select_action(self, x):
        x = x.to(self.device) if x.device != self.device else x  # size([sequence_len, state_dim]) --> size([1, sequence_len, state_dim])
        action_mean = self.actor(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        action = action_mean if self.deterministic else probs.sample()
        return action


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
    def load_checkpoint(self, ckpt_path, evaluate=False, map_location='cuda:0'):
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
