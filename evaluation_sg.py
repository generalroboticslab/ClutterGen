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

from PPO.PPO_continuous_sg import *
from RoboSensai_bullet import *
from multi_envs import *
from utils import *

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate Handem Pushing Experiment')
    parser.add_argument('--rendering', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument("--num_envs", type=int, default=1, help="the number of parallel game environments")
    parser.add_argument('--object_pool_name', type=str, default='Union', help="Object Pool. Ex: YCB, Partnet")
    parser.add_argument('--result_dir', type=str, default='train_res', required=False)
    parser.add_argument('--save_dir', type=str, default='eval_res', required=False)
    parser.add_argument('--collect_data', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--checkpoint', type=str, default='Union_01-15_21:48_Transformer_Tanh_Rand_ObjPlace_QRRegion_Goal_maxObjNum1_maxPool240_maxScene1_maxStable50_contStable20_maxQR1Scene_Epis2Replaceinf_Weight_rewardPobj100.0') # also point to json file path
    parser.add_argument('--index_episode', type=str, default='best')
    parser.add_argument('--eval_result', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
    parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics Device ID')
    parser.add_argument('--num_trials', type=int, default=10000)  # database length if have

    parser.add_argument('--random_policy', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--heuristic_policy', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--generate_benchmark', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--save_to_assets', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--use_benchmark', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    
    parser.add_argument('--draw_contact', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help='Draw contact force direction')
    parser.add_argument('--quiet', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--failure_only', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--success_only', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--discrete_replay', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--realtime', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--real', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument("--deterministic", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="Using deterministic policy instead of normal")

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
    parser.add_argument('--seed', type=int, default=123456, help='Contact force noise range')

    # Granular Media
    parser.add_argument('--add_gms', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Add Granular Media')
    parser.add_argument('--add_sides_shelf', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Add bucket for Granular Media')
    parser.add_argument('--num_gms', type=int, default=1000)

    # RoboSensai Bullet parameters
    parser.add_argument('--asset_root', type=str, default='assets', help="folder path that stores all urdf files")
    parser.add_argument('--object_pool_folder', type=str, default='union_objects_test', help="folder path that stores all urdf files")
    parser.add_argument('--scene_pool_folder', type=str, default='union_scene', help="folder path that stores all urdf files")
    parser.add_argument('--specific_scene', type=str, default=None)

    parser.add_argument('--num_pool_objs', type=int, default=32)
    parser.add_argument('--num_pool_scenes', type=int, default=1)
    parser.add_argument('--max_num_qr_scenes', type=int, default=1) 
    parser.add_argument('-n', '--max_num_placing_objs_lst', type=json.loads, default=list(range(1, 2)), help='A list of max num of placing objs')
    parser.add_argument('--random_select_objs_pool', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help='Draw contact force direction')
    parser.add_argument('--random_select_scene_pool', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Draw contact force direction')
    parser.add_argument('--random_select_placing', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help='Draw contact force direction')
    parser.add_argument('--seq_select_placing', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Draw contact force direction')
    parser.add_argument('--fixed_qr_region', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help='Draw contact force direction')
    parser.add_argument('--fixed_scene_only', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help='Draw contact force direction')
    parser.add_argument('--num_episode_to_replace_pool', type=int, default=np.inf)
    parser.add_argument('--critic_visualize', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Visualize critic')


    eval_args = parser.parse_args()
    eval_args.json_file_path = os.path.join(eval_args.result_dir, eval_args.object_pool_name, 'Json', eval_args.checkpoint+'.json')
    checkpoint_folder = os.path.join(eval_args.result_dir, eval_args.object_pool_name, 'checkpoints', eval_args.checkpoint)
    eval_args.checkpoint_path = os.path.join(checkpoint_folder, eval_args.checkpoint + '_' + eval_args.index_episode)
    
    restored_eval_args = eval_args.__dict__.copy()  # store eval_args to avoid overwrite

    with open(eval_args.json_file_path, 'r') as json_obj: # Read the training args
        args_json = json.load(json_obj)

    # Keep the training args if evaluation args is None

    eval_args.__dict__.update(args_json) # store in train_args
    eval_args.__dict__.update(restored_eval_args) # overwrite by eval_args to become real eval_args

    # real module
    if eval_args.real: eval_args.collect_data = False
    # Draw contact should under the rendering situation
    if eval_args.rendering is False: eval_args.draw_contact = False
    # Critic visualize should under the 1 env rendering situation
    if eval_args.critic_visualize:
        assert eval_args.num_envs == 1, "Only support 1 env for critic visualization for now"
    
    # create result folder
    eval_args.save_dir = os.path.join(eval_args.save_dir, eval_args.object_pool_name)
    if eval_args.collect_data and not os.path.exists(eval_args.save_dir):
        os.makedirs(eval_args.save_dir)

    # assign an uniform name
    ckeckpoint_index = ''
    if eval_args.random_policy: eval_args.final_name = f'EVAL_RandPolicy'
    elif eval_args.heuristic_policy: eval_args.final_name = f'EVAL_HeurPolicy'
    else: ckeckpoint_index = '_EVAL_' + eval_args.index_episode 
    
    obj_range = f'_objRange_{min(eval_args.max_num_placing_objs_lst)}_{max(eval_args.max_num_placing_objs_lst)}'
    temp_filename = eval_args.final_name + ckeckpoint_index + obj_range
    
    maximum_name_len = 250
    if len(temp_filename) > maximum_name_len: # since the name too long error, I need to shorten the training name 
        shorten_name_range = len(temp_filename) - maximum_name_len
        eval_args.final_name = eval_args.final_name[:-shorten_name_range]
    eval_args.final_name = eval_args.final_name + ckeckpoint_index + obj_range

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

    ################ Save Eval Json File ##################
    if eval_args.collect_data:
        with open(eval_args.json_file, 'w') as json_obj:
            json.dump(vars(eval_args), json_obj, indent=4)

    ################ create world and scene set up ##################
    # Create the gym environment
    envs = create_multi_envs(eval_args, 'spawn')
    temp_env = envs.tempENV; tensor_dtype = temp_env.tensor_dtype
    torch.manual_seed(eval_args.seed); np.random.seed(eval_args.seed); random.seed(eval_args.seed)

    # Agent
    agent = None
    if eval_args.eval_result and not eval_args.random_policy and not eval_args.heuristic_policy:
        agent = Agent(envs.tempENV).to(device)
        agent.load_checkpoint(eval_args.checkpoint_path, evaluate=True, map_location="cuda:0")

    # Evaluate checkpoint before replay
    for max_num_placing_objs in eval_args.max_num_placing_objs_lst:
        envs.env_method('set_args', 'max_num_placing_objs', max_num_placing_objs)

        num_episodes = 0 
        episode_rewards = torch.zeros((eval_args.num_envs, ), device=device, dtype=torch.float32)
        episode_timesteps = torch.zeros((eval_args.num_envs, ), device=device, dtype=torch.float32)
        episode_rewards_box = torch.zeros((eval_args.num_trials, ), device=device, dtype=torch.float32)
        episode_success_box = torch.zeros((eval_args.num_trials, ), device=device, dtype=torch.float32)
        success_scene_cfg = []

        if agent is not None:
            with torch.no_grad():
                seq_obs = torch.Tensor(envs.reset()).to(device)
                # Scene and obj feature tensor are keeping updated inplace
                scene_ft_obs = torch.zeros((eval_args.num_envs, ) + (temp_env.scene_ft_dim, ), dtype=tensor_dtype).to(device)
                obj_ft_obs = torch.zeros((eval_args.num_envs, ) + (temp_env.obj_ft_dim, ), dtype=tensor_dtype).to(device)
                agent.preprocess_pc_update_tensor(scene_ft_obs, obj_ft_obs, envs.reset_infos, use_mask=True)
        
        print(f" Start Evaluating: {max_num_placing_objs} Num of Placing Objs | {eval_args.num_trials} Trials")

        start_time = time.time()
        while num_episodes < eval_args.num_trials:
            ################ agent evaluation ################
            if eval_args.random_policy:
                action = (torch.rand((eval_args.num_envs, temp_env.action_shape[1]), device=device) * 2 - 1) * 5
            elif eval_args.heuristic_policy:
                assert not eval_args.fixed_qr_region, "Heuristic policy only support fixed_qr_region"
                action = torch.zeros((eval_args.num_envs, temp_env.action_shape[1]), device=device)
            else:
                with torch.no_grad():
                    action, probs = agent.select_action([seq_obs, scene_ft_obs, obj_ft_obs])
                    if eval_args.critic_visualize and envs.env_method('get_env_attr', "info")[0]['stepping']==1:
                        act_sig_grid_tensor = create_mesh_grid(action_ranges=[(0, 1)]*6, num_steps=[5]*6).to(device)
                        raw_actions = inverse_sigmoid(act_sig_grid_tensor)
                        action_log_prob = probs.log_prob(raw_actions)
                        print(f"Mean and Std: {probs.mean}, {probs.stddev}")
                        envs.env_method('visualize_actor_prob', raw_actions, action_log_prob)
                        
            next_seq_obs, reward, done, infos = envs.step(action)
            if agent is not None:
                agent.preprocess_pc_update_tensor(scene_ft_obs, obj_ft_obs, infos, use_mask=True)
            
            next_seq_obs, done = torch.Tensor(next_seq_obs).to(device), torch.Tensor(done).to(device)
            reward = torch.Tensor(reward).to(device).view(-1) # if reward is not tensor inside

            seq_obs = next_seq_obs
            episode_rewards += reward
            
            terminal_index = done == 1
            terminal_nums = terminal_index.sum().item()
            # Compute the average episode rewards.
            if terminal_nums > 0:
                num_episodes += terminal_nums
                terminal_ids = terminal_index.nonzero().flatten()

                update_tensor_buffer(episode_rewards_box, episode_rewards[terminal_index])
                success_buf = torch.Tensor(combine_envs_float_info2list(infos, 'success', terminal_ids)).to(device)
                update_tensor_buffer(episode_success_box, success_buf)
                steps_buf = torch.Tensor(combine_envs_float_info2list(infos, 'his_steps', terminal_ids)).to(device)
                update_tensor_buffer(episode_timesteps, steps_buf)
                success_ids = terminal_ids[success_buf.to(torch.bool)]
                success_scene_cfg.extend(combine_envs_float_info2list(infos, 'placed_obj_poses', success_ids))

                print_info = f"Episodes: {num_episodes}" + f" / Total Success: {episode_success_box.sum().item()}" 
                if eval_args.num_envs == 1:
                    print_info += f" / Episode reward: {episode_rewards.item()}"
                print(print_info)
                
                episode_rewards[terminal_index] = 0.
                
        episode_reward = torch.mean(episode_rewards_box).item()
        success_rate = torch.mean(episode_success_box).item()
        unstable_steps = torch.mean(episode_timesteps).item()
        machine_time = time.time() - start_time
        
        print(f"Num of Placing Objs: {max_num_placing_objs} | {eval_args.num_trials} Trials | Success Rate: {success_rate * 100}% | Avg Reward: {episode_reward} |", end=' ')
        print(f"Time: {machine_time} | Num of Env: {eval_args.num_envs} | Episode Steps: {unstable_steps}", end='\n\n')
        
        # Save the evaluation result
        if eval_args.collect_data:
            csv_result = {"max_num_placing_objs": max_num_placing_objs, 
                        "num_trials": eval_args.num_trials,
                        "success_rate": success_rate,
                        "unstable_steps": unstable_steps,
                        "avg_reward": episode_reward,
                        "machine_time": machine_time,
                        "success_scene_cfg": success_scene_cfg}
            write_csv_line(eval_args.result_file_path, csv_result)

            # Save success rate and placed objects number
            meta_data = {
                "episode": num_episodes,
                "scene_obj_success_num": combine_envs_dict_info2dict(infos, key="scene_obj_success_num"),
                "obj_success_rate": combine_envs_dict_info2dict(infos, key="obj_success_rate"),
            }
            save_json(meta_data, os.path.join(eval_args.trajectory_dir, "meta_data.json"))

    print('Process Over')
    envs.close()