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
import pickle

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
    parser.add_argument('--checkpoint', type=str, default='Union_03-12_23:40Sync_Beta_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab60_contStab20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial5_entropy0.01_seed123456') # also point to json file path
    parser.add_argument('--index_episode', type=str, default='best')
    parser.add_argument('--eval_result', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
    parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics Device ID')
    parser.add_argument('--num_trials', type=int, default=10000)  # database length if have
    parser.add_argument('--num_success_trials', type=int, default=None)  # database length if have

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
    parser.add_argument('--QueryRegion_pos', type=json.loads, default=None, help='A list of max num of placing objs')
    parser.add_argument('--QueryRegion_euler_z', type=float, default=None, help='A list of max num of placing objs')
    parser.add_argument('--QueryRegion_halfext', type=json.loads, default=None, help='A list of max num of placing objs') # [0.25, 0.25, 0.35] for realrobot
    parser.add_argument('--random_qr_pos', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Add random noise to contact info')
    parser.add_argument('--random_qr_rotz', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Add random noise to contact info')
    parser.add_argument('--random_srk_qr_halfext', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Add random noise to contact info')
    parser.add_argument('--random_exp_qr_halfext', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Add random noise to contact info')

    # RoboSensai Bullet parameters
    parser.add_argument('--scene_pool_folder', type=str, default='tabletop_selected_scene', help="folder path that stores all urdf files")
    parser.add_argument('--specific_scene', type=str, default="table")
    parser.add_argument('--num_pool_objs', type=int, default=10)
    parser.add_argument('--num_pool_scenes', type=int, default=1)
    parser.add_argument('-n', '--max_num_placing_objs_lst', type=json.loads, default=[10], help='A list of max num of placing objs')
    parser.add_argument('--random_select_objs_pool', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Draw contact force direction')
    parser.add_argument('--random_select_scene_pool', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Draw contact force direction')
    parser.add_argument('--random_select_placing', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help='Draw contact force direction')
    parser.add_argument('--seq_select_placing', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Draw contact force direction')
    # parser.add_argument('--fixed_qr_region', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help='Draw contact force direction')
    parser.add_argument("--max_stable_steps", type=int, default=40, help="the maximum steps for the env to be stable considering success")
    parser.add_argument("--min_continue_stable_steps", type=int, default=20, help="the minimum steps that the object needs to keep stable")
    parser.add_argument('--fixed_scene_only', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help='Draw contact force direction')
    parser.add_argument('--num_episode_to_replace_pool', type=int, default=np.inf)
    parser.add_argument('--actor_visualize', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Visualize critic')
    parser.add_argument('--blender_record', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Visualize critic')
    parser.add_argument('--new_tablehalfExtents', type=json.loads, default=None, help='A list of max num of placing objs')
    parser.add_argument('--strict_checking', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Apply a strict stable checker')

    # Downstream task1 stable placement parameters
    parser.add_argument('--sp_data_collection', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='collect stable placement data')
    parser.add_argument('--sp_num_data', type=int, default=10000, help='collect stable placement data for certain episodes')


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
    if eval_args.actor_visualize:
        assert eval_args.num_envs == 1, "Only support 1 env for critic visualization for now"
    
    # create result folder
    eval_args.save_dir = os.path.join(eval_args.save_dir, eval_args.object_pool_name)
    if eval_args.collect_data and not os.path.exists(eval_args.save_dir):
        os.makedirs(eval_args.save_dir)

    # assign an uniform name
    checkpoint_index = ''
    if eval_args.random_policy: eval_args.final_name = f'EVAL_RandPolicy'
    elif eval_args.heuristic_policy: eval_args.final_name = f'EVAL_HeurPolicy'
    else: checkpoint_index = '_EVAL_' + eval_args.index_episode 
    
    if eval_args.new_tablehalfExtents: checkpoint_index += "_TableHalfExtents" + "_".join(map(str, eval_args.new_tablehalfExtents))
    if eval_args.specific_scene: checkpoint_index += "_Scene_" + eval_args.specific_scene
    if eval_args.QueryRegion_pos: checkpoint_index += "_QRPos_" + "_".join(map(str, eval_args.QueryRegion_pos))
    if eval_args.QueryRegion_euler_z: checkpoint_index += "_QREulerZ_" + str(eval_args.QueryRegion_euler_z)
    if eval_args.QueryRegion_halfext: checkpoint_index += "_QRHalfExt_" + "_".join(map(str, eval_args.QueryRegion_halfext))
    if eval_args.random_qr_pos: checkpoint_index += "_RQRPos"
    if eval_args.random_qr_rotz: checkpoint_index += "_RQREulerZ"
    if eval_args.random_srk_qr_halfext: checkpoint_index += "_RsrkQRHalfExt"
    if eval_args.random_exp_qr_halfext: checkpoint_index += "_RexpQRHalfExt"
    obj_range = f'_objRange_{min(eval_args.max_num_placing_objs_lst)}_{max(eval_args.max_num_placing_objs_lst)}'
    temp_filename = eval_args.final_name + checkpoint_index + obj_range
    
    maximum_name_len = 250
    if len(temp_filename) > maximum_name_len: # since the name too long error, I need to shorten the training name 
        shorten_name_range = len(temp_filename) - maximum_name_len
        eval_args.final_name = eval_args.final_name[:-shorten_name_range]
    eval_args.final_name = eval_args.final_name + checkpoint_index + obj_range

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

    if eval_args.collect_data and eval_args.eval_result:
        check_file_exist(eval_args.result_file_path)

    if eval_args.save_to_assets:  # directly save new assets file to a_new_assets
        if eval_args.use_benchmark: raise Exception("Can not set save_to_assets and use_benchmark to be both true at the same time!") # can not
        if eval_args.collect_data or eval_args.generate_benchmark:
            assets_dir = os.path.split(eval_args.save_to_assets_path)[0]
            if not os.path.exists(assets_dir): os.mkdir(assets_dir)
            check_file_exist(eval_args.save_to_assets_path)

    # create trajectory folder
    eval_args.trajectory_dir = os.path.join(eval_args.save_dir, 'trajectories', eval_args.final_name)
    if eval_args.collect_data and eval_args.eval_result and os.path.exists(eval_args.trajectory_dir): shutil.rmtree(eval_args.trajectory_dir)
    if eval_args.collect_data and not os.path.exists(eval_args.trajectory_dir):
        os.makedirs(eval_args.trajectory_dir)

    # create blender folder
    eval_args.blender_dir = os.path.join(eval_args.save_dir, 'blender', eval_args.final_name)
    if eval_args.collect_data and eval_args.eval_result and os.path.exists(eval_args.blender_dir): 
        shutil.rmtree(eval_args.blender_dir)
    if eval_args.collect_data and eval_args.blender_record and not os.path.exists(eval_args.blender_dir):
        os.makedirs(eval_args.blender_dir)

    # create json folder
    eval_args.json_dir = os.path.join(eval_args.save_dir, 'Json')
    if eval_args.collect_data and not os.path.exists(eval_args.json_dir):
        os.makedirs(eval_args.json_dir)
    eval_args.json_file = os.path.join(eval_args.json_dir, eval_args.final_name + '.json')

    # create SP dataset folder
    eval_args.sp_dataset_dir = os.path.join("StablePlacement", 'SP_Dataset')
    sp_dataset_name = f"{eval_args.specific_scene}_{max(eval_args.max_num_placing_objs_lst)}_{os.path.basename(eval_args.object_pool_folder)}"
    eval_args.sp_dataset_path = os.path.join(eval_args.sp_dataset_dir, sp_dataset_name + '.h5')
    if eval_args.collect_data and eval_args.sp_data_collection:
        os.makedirs(eval_args.sp_dataset_dir, exist_ok=True)
        check_file_exist(eval_args.sp_dataset_path)

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
    # env and scene setup
    if eval_args.num_envs >1:
        envs = create_multi_envs(eval_args, 'forkserver')
        temp_env = envs.tempENV; tensor_dtype = temp_env.tensor_dtype
    elif eval_args.num_envs == 1:
        envs = RoboSensaiBullet(eval_args)
        temp_env = envs; tensor_dtype = temp_env.tensor_dtype
    torch.manual_seed(eval_args.seed); np.random.seed(eval_args.seed); random.seed(eval_args.seed)

    # Agent
    agent = None
    if eval_args.eval_result and not eval_args.random_policy and not eval_args.heuristic_policy:
        agent = Agent(temp_env).to(device)
        agent.load_checkpoint(eval_args.checkpoint_path, evaluate=True, map_location="cuda:0")

    sp_data_index = 0
    if eval_args.num_success_trials is not None or \
      (eval_args.sp_data_collection and eval_args.sp_num_data is not None):
        eval_args.num_trials = int(1e5) # Set a large number to collect data
    
    ################ Evaluate checkpoint before replay ##################
    for max_num_placing_objs in eval_args.max_num_placing_objs_lst:
        if eval_args.num_envs > 1:
            envs.env_method('set_args', 'max_num_placing_objs', max_num_placing_objs)
        else:
            envs.args.max_num_placing_objs = max_num_placing_objs

        num_episodes = 0
        episode_rewards = torch.zeros((eval_args.num_envs, ), device=device, dtype=torch.float32)
        episode_rewards_box = torch.zeros((eval_args.num_trials, ), device=device, dtype=torch.float32)
        episode_success_box = torch.zeros((eval_args.num_trials, ), device=device, dtype=torch.float32)
        obj_stable_steps_box = []
        scene_cfg_dict = {}; success_scene_cfg_dict = {}; actor_traj_log_dict = {}; 
        placement_traj_dict = {}; qr_region_dict = {}
        for i in range(eval_args.num_envs):
            actor_traj_log_dict[i] = {
                "prob_pos_heatmap": [],
                "probs_mean_std": [],
            }

        if agent is not None:
            with torch.no_grad():
                next_seq_obs = torch.Tensor(envs.reset()).to(device)
                # Scene and obj feature tensor are keeping updated inplace
                next_scene_ft_obs = torch.zeros((eval_args.num_envs, ) + (temp_env.scene_ft_dim, ), dtype=tensor_dtype).to(device)
                next_obj_ft_obs = torch.zeros((eval_args.num_envs, ) + (temp_env.obj_ft_dim, ), dtype=tensor_dtype).to(device)
                reset_infos = envs.reset_infos if eval_args.num_envs > 1 else [envs.info]
                agent.preprocess_pc_update_tensor(next_scene_ft_obs, next_obj_ft_obs, reset_infos, use_mask=True)
        
        print(f" Start Evaluating: {max_num_placing_objs} Num of Placing Objs | {eval_args.num_trials} Trials | {eval_args.num_success_trials} Success Trials Required")

        start_time = time.time()
        while num_episodes < eval_args.num_trials:
            if eval_args.sp_data_collection and \
               eval_args.sp_num_data is not None and \
               sp_data_index >= eval_args.sp_num_data:
                break

            ################ agent evaluation ################
            if eval_args.random_policy or eval_args.heuristic_policy:
                action = torch.rand((eval_args.num_envs, temp_env.action_shape[1]), device=device)
            else:
                with torch.no_grad():
                    action, probs = agent.select_action([next_seq_obs, next_scene_ft_obs, next_obj_ft_obs])
                    
                    if eval_args.actor_visualize:
                        std_range = 2 # +- 2 std range
                        probs_mean, probs_std = probs.mean, probs.stddev
                        raw_act_range_low = (probs_mean - std_range*probs_std).squeeze(dim=0)
                        raw_act_range_high = (probs_mean + std_range*probs_std).squeeze(dim=0)
                        act_range_low, act_range_high = torch.clamp(raw_act_range_low, 0., 1.), torch.clamp(raw_act_range_high, 0., 1.)
                        action_ranges = list(zip(act_range_low, act_range_high))
                        act_sig_grid_tensor = create_mesh_grid(action_ranges=action_ranges, num_steps=[5]*len(action_ranges)).to(device)
                        action_log_prob = probs.log_prob(act_sig_grid_tensor)

                        prob_pos_heatmap = envs.visualize_actor_prob(act_sig_grid_tensor, action_log_prob, action)

                        frame_index = None
                        if eval_args.blender_record:
                            frame_index = envs.pybullet_recorder.frame_index

                        actor_traj_log_dict[0]["prob_pos_heatmap"].append((frame_index, prob_pos_heatmap))
                        actor_traj_log_dict[0]["probs_mean_std"].append((frame_index, probs.mean.cpu().numpy(), probs.stddev.cpu().numpy()))
                        
            next_seq_obs, reward, done, infos = envs.step(action)
            if agent is not None:
                agent.preprocess_pc_update_tensor(next_scene_ft_obs, next_obj_ft_obs, infos, use_mask=True)
            
            next_seq_obs, done = torch.Tensor(next_seq_obs).to(device), torch.Tensor(done).to(device)
            reward = torch.Tensor(reward).to(device).view(-1) # if reward is not tensor inside

            episode_rewards += reward
            
            terminal_index = done == 1
            terminal_nums = terminal_index.sum().item()
            # Compute the average episode rewards.
            if terminal_nums > 0:
                terminal_ids = terminal_index.nonzero().flatten()
                update_tensor_buffer(episode_rewards_box, episode_rewards[terminal_index])
                success_buf = torch.Tensor(combine_envs_float_info2list(infos, 'success', terminal_ids)).to(device)
                update_tensor_buffer(episode_success_box, success_buf)
                success_ids = terminal_ids[success_buf.to(torch.bool)]
                
                if eval_args.collect_data:
                    scene_cfg = combine_envs_float_info2list(infos, 'placed_obj_poses', terminal_ids)
                    placement_trajs = combine_envs_float_info2list(infos, "placement_trajs", terminal_ids)
                    qr_regions = combine_envs_float_info2list(infos, 'qr_region', terminal_ids)
                    for i, env_id in enumerate(terminal_ids):
                        scene_cfg_dict.update({num_episodes + i: scene_cfg[i]})
                        placement_traj_dict.update({num_episodes + i: [placement_trajs[i], success_buf[i].item()]})
                        qr_region_dict.update({num_episodes + i: qr_regions[i]})
                        if env_id in success_ids:
                            success_scene_cfg_dict.update({num_episodes + i: scene_cfg[i]})
                        
                        for obj_name in placement_trajs[i].keys():
                            obj_stable_steps_sum = sum(placement_trajs[i][obj_name]['stable_steps'])
                            obj_stable_steps_box.append(obj_stable_steps_sum)

                        if eval_args.actor_visualize:
                            success_suffix = 'success' if env_id in success_ids else 'failure'
                            path = os.path.join(eval_args.trajectory_dir, f"{max_num_placing_objs}Objs_{num_episodes + i}eps_{success_suffix}_actor_traj_log.pkl")
                            pickle.dump(actor_traj_log_dict[env_id.item()], open(path, 'wb'))
                            actor_traj_log_dict[env_id.item()] = {
                                "prob_pos_heatmap": [],
                                "probs_mean_std": [],
                            }
                        
                    if eval_args.sp_data_collection and len(success_ids) > 0:
                        sp_dataset_lst = combine_envs_float_info2list(infos, 'sp_dataset', success_ids)
                        with h5py.File(eval_args.sp_dataset_path, 'a') as f:
                            for sp_data in sp_dataset_lst:
                                if not sp_data:
                                    continue
                                for index, single_sp_data_point in sp_data.items():
                                    group = create_or_update_group(f, f"{sp_data_index}")
                                    create_dataset_recursively(group, single_sp_data_point)
                                    sp_data_index += 1

                if eval_args.blender_record:
                    blender_recorders_lst = combine_envs_float_info2list(infos, 'blender_recorder', terminal_ids)
                    for i, blender_recorder in enumerate(blender_recorders_lst):
                        index_suffix = f"{num_episodes + i}"
                        success_suffix = 'success' if success_buf[i].item() else 'failure'
                        if eval_args.collect_data:
                            blender_recorder.save(os.path.join(eval_args.blender_dir, 
                                                  f"{max_num_placing_objs}Objs_{index_suffix}eps_{success_suffix}_blender.pkl"))

                num_episodes += terminal_nums
                print_info = f"Episodes: {num_episodes}" + f" / Total Success: {episode_success_box.sum().item()}" 
                if eval_args.num_envs == 1:
                    print_info += f" / Episode reward: {episode_rewards.item()}"
                print(print_info)
                
                episode_rewards[terminal_index] = 0.
                if eval_args.num_success_trials is not None \
                    and episode_success_box[-num_episodes:].sum().item() >= eval_args.num_success_trials:
                    break
                
        episode_reward = torch.mean(episode_rewards_box[-num_episodes:]).item()
        success_rate = torch.mean(episode_success_box[-num_episodes:]).item()
        obj_stable_steps_mean, obj_stable_steps_std = np.mean(obj_stable_steps_box), np.std(obj_stable_steps_box)
        machine_time = time.time() - start_time
        
        print(f"Num of Placing Objs: {max_num_placing_objs} | {eval_args.num_trials} Trials | Success Rate: {success_rate * 100}% | Avg Stable Steps: {obj_stable_steps_mean}, std: {obj_stable_steps_std} | Avg Reward: {episode_reward} |", end=' ')
        print(f"Time: {machine_time} | Num of Env: {eval_args.num_envs}", end='\n\n')
        
        # Save the evaluation result
        if eval_args.collect_data:
            csv_result = {
                "max_num_placing_objs": max_num_placing_objs, 
                "num_trials": eval_args.num_trials,
                "success_rate": success_rate,
                "avg_reward": episode_reward,
                "obj_stable_steps_mean": obj_stable_steps_mean,
                "obj_stable_steps_std": obj_stable_steps_std,
                "machine_time": machine_time
            }
            write_csv_line(eval_args.result_file_path, csv_result)
            print(f"Saved evaluation CSV to {eval_args.result_file_path}")

            # Save success rate and placed objects number
            meta_data = {
                "episode": num_episodes,
                "scene_obj_success_num": combine_envs_dict_info2dict(infos, key="scene_obj_success_num"),
                "qr_scene_pose": combine_envs_float_info2list(infos, 'qr_scene_pose')[0],
                "obj_success_rate": combine_envs_dict_info2dict(infos, key="obj_success_rate"),
                "scene_cfgs": scene_cfg_dict,
                "success_scene_cfgs": success_scene_cfg_dict,
                "placement_trajs": placement_traj_dict,
                "qr_region_dict": qr_region_dict
            }
            save_json(meta_data, os.path.join(eval_args.trajectory_dir, f"{max_num_placing_objs}Objs_meta_data.json"))

    if eval_args.sp_data_collection:
        print(f"Saved stable placement dataset to {eval_args.sp_dataset_path}; Num Data Points: {sp_data_index}")
                
    print('Process Over')
    envs.close()