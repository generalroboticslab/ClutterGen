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

from gym_env import *
import copy
import misc_utils as mu

HIDDEN_SIZE = 64
MIN_STD = 0.05
INITIAL_STD = 1

def make_env(global_args, seed, idx, capture_video, run_name):
    args = copy.deepcopy(global_args); args.seed = seed # copy args and change its seed to guanrantee each environment has different seed values
    scene_set_up = get_scene_set_up(args)

    def thunk():
        dynamic_grasping_world = create_grasping_world(args)

        ####### Define actions dim #######
        if args.action_type == 'budget_discrete': actions_dim_list = [-2, len(args.fix_motion_planning_time)] # prediction_horizon is None; use hard computation
        elif args.action_type == 'prediction_discrete': actions_dim_list = [len(args.fix_prediction_time), 3] # current best time_budget is only 3
        elif args.action_type == 'both': actions_dim_list = [len(args.fix_prediction_time), len(args.fix_motion_planning_time)]
        elif args.action_type in ['both_continuous', 'budget_continuous', 'prediction_continuous']: actions_dim_list = [[min(args.fix_prediction_time), min(args.fix_motion_planning_time)], [max(args.fix_prediction_time), max(args.fix_motion_planning_time)]]
        else: raise Exception("Invalid action_type; must in ['budget_discrete', 'prediction_discrete', 'both_continuous', 'budget_continuous', 'prediction_continuous']")

        env = DynamicGraspRL(grasp_world=dynamic_grasping_world, scene_set_up=scene_set_up, args=args, sequence_len=args.sequence_len,
                             reward_scale=args.assigned_reward, action_type=args.action_type, seed=args.seed,
                             actions_dim_list=actions_dim_list)  # used if action type is budget_discrete

        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class LSTM_Linear(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_first=True, init_std=0.01):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.linear = layer_init(nn.Linear(hidden_size, output_size), std=init_std)

    def forward(self, x):
        sequence_features, (hn, cn) = self.lstm(x) # hn -> hidden state; cn -> cell state
        output = self.linear(sequence_features[:, -1, :]) # only use the last layer output
        return output

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.envs = envs
        self.action_type = 'both_continous' if len(envs.action_space.shape)==1 else 'single_ablation'
        self.use_lstm = envs.use_lstm
        self.min_std = MIN_STD; self.cur_std = INITIAL_STD # torch.exp(torch.zeros())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not self.use_lstm: # Use MLP
            self.critic = nn.Sequential(
                layer_init(nn.Linear(np.prod(envs.observation_space.shape), HIDDEN_SIZE)),
                nn.Tanh(),
                layer_init(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)),
                nn.Tanh(),
                layer_init(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)),
                nn.Tanh(),
                layer_init(nn.Linear(HIDDEN_SIZE, 1), std=1.0),
            ).to(self.device)
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(np.prod(envs.observation_space.shape), HIDDEN_SIZE)),
                nn.Tanh(),
                layer_init(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)),
                nn.Tanh(),
                layer_init(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)),
                nn.Tanh(),
                layer_init(nn.Linear(HIDDEN_SIZE, np.prod(envs.action_space.shape)), std=0.01),
            ).to(self.device)
        else: # Use LSTM and 1 MLP
            self.critic = LSTM_Linear(input_size=envs.observation_space.shape[1], hidden_size=HIDDEN_SIZE,
                                      num_layers=envs.observation_space.shape[0], output_size=1, batch_first=True, init_std=1.0).to(self.device)
            self.actor_mean = LSTM_Linear(input_size=envs.observation_space.shape[1], hidden_size=HIDDEN_SIZE,
                                      num_layers=envs.observation_space.shape[0], output_size=np.prod(envs.action_space.shape), batch_first=True, init_std=0.01).to(self.device)

        # Not train standard deviation, but only use linear schedular
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space.shape)).to(self.device), requires_grad=False) # is_leaf problem for nn.parameters/must set to() within it

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        cov_mat = torch.diag_embed(action_std)
        probs = MultivariateNormal(action_mean, cov_mat)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def select_action(self, state, evaluate=True):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)  # [1, 2, 3] --> [[1, 2, 3]]
        action_mean = self.actor_mean(state)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        cov_mat = torch.diag_embed(action_std) # implicit assume that prediction_horizon and time_budget are independent
        probs = MultivariateNormal(action_mean, cov_mat)
        action = probs.sample().flatten()
        return action.cpu().tolist()

    def decay_std(self, decay_value=0.05):
        self.cur_std = max(self.cur_std-decay_value, self.min_std)
        self.actor_logstd = nn.Parameter(torch.full(self.actor_logstd.size(), np.log(self.cur_std)).to(self.device), requires_grad=False)

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
    def load_checkpoint(self, ckpt_path, evaluate=False, load_memory=True):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.load_state_dict(checkpoint)

            if evaluate: self.set_mode('eval')
            else: self.set_mode('train')

"""
    # result = [('exp_idx', i_episode),
    #           ('success', success),
    #           ('Fixed_motion_planner_time', episode_actions),
    #           ('Real_motion_planner_time', real_motion_planning_time),
    #           ('dynamic_grasping_time', dynamic_grasping_time),
    #
    #           ('grasp_idx', grasp_idx),
    #           ('grasp_switched_list', grasp_switched_list),
    #           ('num_ik_called_list', num_ik_called_list)]
    #
    # mu.write_csv_line(result_file_path=args.result_file_path, result=result)
"""
def ppo_record(infos, episode_rewards, episode_actions, args, i_episode): # record training results
    cur_success_list = []; cur_reward_list = []; cur_action_prediction = []; cur_action_budget = [] # success_all also equals to episodes number
    terminal_index = []
    for idx, others in enumerate(infos):
        if others == 0: continue
        else:
            # whether store data or not
            terminal_index.append(idx)  # record which environment episode is done
            success, grasp_idx, dynamic_grasping_time, grasp_switched_list, num_ik_called_list, real_motion_planning_time = others['others']
            # append one env success
            cur_success_list.append(success)
            # compute one env reward average
            cur_reward_list.append(np.sum(episode_rewards[idx]) * args.gamma ** len(episode_rewards[idx]))
            episode_rewards[idx].clear() # clean the terminated episode row reward
            # compute one env action average
            mat_v_actions = np.array(episode_actions[idx])
            cur_action_prediction.append(np.mean(mat_v_actions[:, 0]))
            cur_action_budget.append(np.mean(mat_v_actions[:, 1]))
            episode_actions[idx].clear()
    return cur_success_list, cur_reward_list, np.mean(cur_action_prediction), np.mean(cur_action_budget), terminal_index

def compute_mean_multi(episode_box, terminal_index):
    res = []
    for idx in terminal_index:
        res.append(np.mean(episode_box[idx]))
        episode_box[idx] = []  # clean the terminated episode row actions/reward
    if res: return np.mean(res) # return a mean value
    else : return 0

def control_len(lst, length=100): # control a list length to be a certain number
    if len(lst) <= length: return lst
    else: return lst[len(lst)-length:]

if __name__ == "__main__":

    class TestModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor_std = nn.Parameter(torch.zeros(1, 2), requires_grad=False)
