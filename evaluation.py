import argparse
import datetime
import numpy as np
import wandb
import json
import psutil
from collections import deque
import shutil

import os
import random
import time
from distutils.util import strtobool

from isaacgym.gymutil import parse_device_str
from PPO.PPO_discrete_vit import *
from RoboSensai_env import *

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate Handem Pushing Experiment')
    parser.add_argument('--rendering', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument("--num_envs", type=int, default=1, help="the number of parallel game environments")
    parser.add_argument('--object_name', type=str, default='cube', help="Target object to be grasped. Ex: cube")
    parser.add_argument('--specific_target', type=str, default=None) # the real grasp object
    parser.add_argument('--result_dir', type=str, default='train_res', required=False)
    parser.add_argument('--save_dir', type=str, default='eval_res', required=False)
    parser.add_argument('--collect_data', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--checkpoint', type=str, default='cube_10-06_18:52_Pushing_Transformer_GPU_Rand_goal_targInit_With_Use_Add_limitws_Weight_pos40.0') # also point to json file path
    parser.add_argument('--index_episode', type=str, default='570037')
    parser.add_argument('--eval_result', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
    parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics Device ID')

    parser.add_argument('--num_trials', type=int, default=10000)  # database length if have
    parser.add_argument('--use_benchmark', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--save_to_assets', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--random_policy', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--generate_benchmark', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    
    parser.add_argument('--draw_contact', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help='Draw contact force direction')
    parser.add_argument('--quiet', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--failure_only', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--success_only', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--discrete_replay', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--realtime', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--real', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)

    parser.add_argument('--record_trajectory', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--replay', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--record_video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--record_frames', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--virtual_screen_capture', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--record_real_time', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--prefix', type=str, default='', help="Object to tests on for real experiments. This is for video prefix")
    
    # Evaluation task parameters
    parser.add_argument('--random_target_init', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Randomize goal pose')
    parser.add_argument('--random_goal', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help='Randomize goal pose')
    parser.add_argument('--add_random_noise', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Add random noise to contact info')
    parser.add_argument('--contact_noise_v', type=float, default=0.01, help='Contact position noise range')
    parser.add_argument('--force_noise_v', type=float, default=0.0, help='Contact force noise range')

    # Granular Media
    parser.add_argument('--add_gms', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Add Granular Media')
    parser.add_argument('--add_sides_shelf', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Add bucket for Granular Media')
    parser.add_argument('--num_gms', type=int, default=1000)

    eval_args = parser.parse_args()
    eval_args.json_file_path = os.path.join(eval_args.result_dir, eval_args.object_name, 'Json', eval_args.checkpoint+'.json')
    checkpoint_folder = os.path.join(eval_args.result_dir, eval_args.object_name, 'checkpoints', eval_args.checkpoint)
    eval_args.checkpoint_path = os.path.join(checkpoint_folder, eval_args.checkpoint + '_' + eval_args.index_episode)
    
    # Isaac Gym operations
    eval_args.sim_device_type, eval_args.compute_device_id = parse_device_str(eval_args.sim_device)
    eval_args.use_gpu = (eval_args.sim_device_type == 'cuda')
    
    restored_eval_args = eval_args.__dict__.copy()  # store eval_args to avoid overwrite

    with open(eval_args.json_file_path, 'r') as json_obj:
        args_json = json.load(json_obj)
    eval_args.__dict__.update(args_json) # store in train_args
    eval_args.__dict__.update(restored_eval_args) # overwrite by eval_args to become real eval_args

    # real module
    if eval_args.real: eval_args.collect_data = False
    # Draw contact should under the rendering situation
    if eval_args.rendering is False: eval_args.draw_contact = False
    # Task maybe not exsit in the old checkpoint
    eval_args.task = eval_args.task if hasattr(eval_args, "task") else "P2G"
    # handed_search may not exsit in the old checkpoint
    eval_args.handed_search = eval_args.handed_search if hasattr(eval_args, "handed_search") else True

    # replace to another object
    # eval_args.target_list = ["tomato_soup_can", "bleach_cleanser", "cube", "mustard_bottle", "potted_meat_can", "power_drill", "sugar_box"]
    if eval_args.specific_target is not None:
        # assert eval_args.specific_target in eval_args.target_list, f"Current object name {eval_args.specific_target} is not in {eval_args.target_list}"
        eval_args.object_name = eval_args.specific_target
        eval_args.random_target = False
    
    # create result folder
    eval_args.save_dir = os.path.join(eval_args.save_dir, eval_args.object_name)
    if eval_args.collect_data and not os.path.exists(eval_args.save_dir):
        os.makedirs(eval_args.save_dir)

    # assign an uniform name
    ckeckpoint_index = ''
    if eval_args.random_policy: eval_args.final_name = f'EVAL_RandomPolicy'
    else: ckeckpoint_index = '_EVAL' + eval_args.index_episode 
    
    eval_args.scene_suffix = '_Setup'
    if eval_args.specific_target is not None: eval_args.scene_suffix += f'_fix_{eval_args.specific_target}'
    temp_filename = eval_args.final_name + ckeckpoint_index + eval_args.scene_suffix
    
    maximum_name_len = 250
    if len(temp_filename) > maximum_name_len: # since the name too long error, I need to shorten the training name 
        shorten_name_range = len(temp_filename) - maximum_name_len
        eval_args.final_name = eval_args.final_name[:-shorten_name_range]
    eval_args.final_name = eval_args.final_name + ckeckpoint_index + eval_args.scene_suffix

    # Generate benchmark table does not use collect_data
    if eval_args.generate_benchmark:
        eval_args.collect_data = False

    # Use uniform name for CSV, Json, and Trajectories name
    print('Uniform name is:', eval_args.final_name)

    # create a global csv file
    eval_args.global_res_dir = os.path.join(eval_args.save_dir, 'Global_Res.csv')

    # create csv folder
    eval_args.csv_dir = os.path.join(eval_args.save_dir, 'CSV')
    if eval_args.collect_data and not os.path.exists(eval_args.csv_dir):
        os.makedirs(eval_args.csv_dir)
    eval_args.result_file_path = os.path.join(eval_args.csv_dir, eval_args.final_name + '.csv')

    # A new asset file name
    eval_args.save_to_assets_path = eval_args.baseline_experiment_path.replace('.csv', f'{eval_args.scene_suffix}.csv')

    if eval_args.collect_data and eval_args.eval_result and os.path.exists(eval_args.result_file_path):
        response = input(f"Find existing result file {eval_args.result_file_path}! Whether remove or not (y/n):")
        if response == 'y' or response == 'Y': os.remove(eval_args.result_file_path)
        else: raise Exception("Give up this evaluation because of exsiting evluation result.")
    if eval_args.save_to_assets:  # directly save new assets file to a_new_assets
        if eval_args.use_benchmark: raise Exception("Can not set save_to_assets and use_benchmark to be both true at the same time!") # can not
        if eval_args.collect_data or eval_args.generate_benchmark:
            assets_dir = os.path.split(eval_args.save_to_assets_path)[0]
            if not os.path.exists(assets_dir): os.mkdir(assets_dir)
            if os.path.exists(eval_args.save_to_assets_path):
                response = input(f"Find existing baseline experiment file {eval_args.save_to_assets_path}! Whether remove or not (y/n):")
                if response == 'y' or response == 'Y': os.remove(eval_args.save_to_assets_path)
                else: raise Exception("Give up this evaluation because of exsiting baseline experiment file but give up overwritting.")

    # create trajectory folder
    eval_args.trajectory_dir = os.path.join(eval_args.save_dir, 'trajectories', eval_args.final_name)
    if eval_args.collect_data and eval_args.eval_result and os.path.exists(eval_args.trajectory_dir): shutil.rmtree(eval_args.trajectory_dir)
    if eval_args.collect_data and not os.path.exists(eval_args.trajectory_dir):
        os.makedirs(eval_args.trajectory_dir)

    # create json folder
    eval_args.json_dir = os.path.join(eval_args.save_dir, 'Json')
    if eval_args.collect_data and not os.path.exists(eval_args.json_dir):
        os.makedirs(eval_args.json_dir)
    eval_args.json_file = os.path.join(eval_args.json_dir, eval_args.final_name + '.json')

    return eval_args

if __name__ == "__main__":
    cur_pid = os.getpid(); print(f"###### Evaluation PID is {cur_pid} ######")
    eval_args = get_args()
    random.seed(eval_args.seed)
    np.random.seed(eval_args.seed)
    torch.manual_seed(eval_args.seed)
    torch.cuda.manual_seed_all(eval_args.seed)
    torch.backends.cudnn.deterministic = eval_args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and eval_args.cuda else "cpu")

    ################ create world and scene set up ##################
    # Create the gym environment
    envs = RoboSensaiEnv(eval_args)
    # torch.manual_seed(eval_args.seed); np.random.seed(eval_args.seed); random.seed(eval_args.seed)

    # Agent
    agent = None
    if eval_args.eval_result and not eval_args.random_policy:
        agent = Agent(envs).to(device)
        agent.load_checkpoint(eval_args.checkpoint_path, load_memory=True, map_location="cuda:0")

    # Evaluate checkpoint before replay
    avg_reward = 0.; num_success = 0; num_episodes = 0; iterations = 0; 
    episode_rewards = torch.zeros((eval_args.num_envs, ), device=envs.device, dtype=torch.float32)
    episode_rewards_box = torch.zeros((eval_args.num_trials, ), device=envs.device, dtype=torch.float32)
    episode_success_box = torch.zeros((eval_args.num_trials, ), device=envs.device, dtype=torch.float32)
    done = torch.zeros(eval_args.num_envs, device=envs.device, dtype=torch.float32)

    if agent is not None: agent.set_mode('eval')  # to avoid batch_norm error if used in the future
    
    with torch.no_grad():
        state_dict = envs.reset()
    print('Evaluating:', f'{eval_args.num_trials} trials') # start evaluation

    while num_episodes < eval_args.num_trials:
        if hasattr(envs, "viewer") and gym.query_viewer_has_closed(envs.viewer):
            break
        ################ agent evaluation ################
        if eval_args.random_policy:
            action = envs.random_actions()
        else:
            vis_obs, vec_obs = state_dict['image'], state_dict['proprioception']
            vis_obs = torch.Tensor(vis_obs).to(device)
            vec_obs = torch.Tensor(vec_obs).to(device)
            action = agent.select_action((vis_obs, vec_obs))
        next_state_dict, reward, done, infos = envs.step(action)

        state_dict = next_state_dict
        iterations += 1
        # print(f'Iteration: {iterations}; Reward: {reward}')
    
        pos_reward, act_penalty = infos['pos_reward'], infos['act_penalty']

        episode_rewards += reward
        
        terminal_index = done == 1
        terminal_nums = terminal_index.sum().item()
        # Compute the average episode rewards.
        if terminal_nums > 0:
            num_episodes += terminal_nums

            update_tensor_buffer(episode_rewards_box, episode_rewards[terminal_index])
            update_tensor_buffer(episode_success_box, envs.success_buf[terminal_index])

            print_info = f"Episodes: {num_episodes}" + f" / Total Success: {episode_success_box.sum().item()}" 
            if eval_args.num_envs == 1:
                print_info += f" / Episode reward: {episode_rewards.item()}"
            print(print_info)
            
            episode_rewards[terminal_index] = 0.
                

    episode_reward = torch.mean(episode_rewards_box).item()
    episode_success_rate = torch.mean(episode_success_box).item()

        # if eval_args.collect_data: # save_file
        #     with open(os.path.join(eval_args.trajectory_dir, 'index-{0}.json'.format(trial)), 'w') as outfile:
        #         json.dump(envs.grasp_world.log_trajectory, outfile)
    
    print(f"{eval_args.num_trials} Trials| Success Rate: {episode_success_rate * 100}% | Avg Reward: {episode_reward}")

    print('Process Over')
    envs.close()


# for i in range(1, 10001, 1000):
#     new_state = torch.rand((i, 10, 19), device='cuda')
#     start_time = time.time()
#     cc = agent.get_action_and_value(new_state)
#     print(f"Number of obs: {i} / Time: {time.time() - start_time}")