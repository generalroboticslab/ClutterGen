# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import argparse
import datetime
import numpy as np
import wandb
import json
import psutil

import os
import random
import time
from distutils.util import strtobool

from RoboSensai_bullet import *
from PPO.PPO_continuous_sg import *

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from multi_envs import *
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='Train Tactile Pushing Experiment')
    
    # Env hyper parameters
    parser.add_argument('--collect_data', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True) # https://docs.python.org/3/library/argparse.html#:~:text=%27%3F%27.%20One%20argument,to%20illustrate%20this%3A
    parser.add_argument('--object_pool_name', type=str, default='Union', help="Target object to be grasped. Ex: cube")
    parser.add_argument('--rendering', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--realtime', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--quiet', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)

    # RoboSensai Env parameters (dataset)
    parser.add_argument('--num_pool_objs', type=int, default=10)
    parser.add_argument('--min_num_placing_objs', type=int, default=1)
    parser.add_argument('--train_step', type=int, default=1)
    parser.add_argument('--max_num_placing_objs', type=int, default=10)
    parser.add_argument('--random_select_objs_pool', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Draw contact force direction')
    parser.add_argument('--random_select_placing', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help='Draw contact force direction')
    parser.add_argument('--num_pool_scenes', type=int, default=1)
    parser.add_argument('--specific_scene', type=str, default="table") 
    parser.add_argument('--random_select_scene_pool', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Draw contact force direction')
    parser.add_argument('--fixed_scene_only', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help='Draw contact force direction')
    parser.add_argument('--fixed_qr_region', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Draw contact force direction')
    parser.add_argument('--num_episode_to_replace_pool', type=int, default=np.inf)
    parser.add_argument('--max_num_urdf_points', type=int, default=1024)
    parser.add_argument('--max_num_scene_points', type=int, default=20480)
    # RoboSensai Env parameters (training)
    parser.add_argument('--max_trials', type=int, default=5)  # maximum steps trial for one object per episode
    parser.add_argument('--max_traj_history_len', type=int, default=240) 
    parser.add_argument("--max_stable_steps", type=int, default=60, help="the maximum steps for the env to be stable considering success")
    parser.add_argument("--min_continue_stable_steps", type=int, default=20, help="the minimum steps that the object needs to keep stable")
    parser.add_argument('--reward_pobj', type=float, default=100., help='Position reward weight')
    parser.add_argument('--penalty', type=float, default=0., help='Action penalty weight')
    parser.add_argument('--vel_reward_scale', type=float, default=0.005, help='scaler for the velocity reward')
    parser.add_argument('--vel_threshold', type=float, default=[0.005, np.pi/360], nargs='+')
    parser.add_argument('--acc_threshold', type=float, default=[1., np.pi], nargs='+') 
    parser.add_argument('--use_bf16', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='default data type')
    parser.add_argument('--use_curriculum', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help='Use curriculum learning')
    parser.add_argument('--patience_iters', type=int, default=5000)

    # I/O hyper parameter
    parser.add_argument('--asset_root', type=str, default='assets', help="folder path that stores all urdf files")
    parser.add_argument('--object_pool_folder', type=str, default='group_objects/group0_dinning_table', help="folder path that stores all urdf files")
    parser.add_argument('--scene_pool_folder', type=str, default='union_scene', help="folder path that stores all urdf files")

    parser.add_argument('--debug', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--result_dir', type=str, default='train_res', required=False)
    parser.add_argument('--wandb', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--force_name', default=None, type=str)

    # Algorithm specific arguments
    parser.add_argument('--env_name', default="RoboSensai_SG", help='Wandb config name')
    parser.add_argument("--use_lstm", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="Toggles whether or not to use LSTM version of meta-controller.")
    parser.add_argument("--use_traj_encoder", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="Toggles whether or not to use Transformer version of meta-controller.")
    parser.add_argument("--use_seq_obs_encoder", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="Toggles whether or not to use Transformer version of meta-controller.")
    parser.add_argument("--use_tf_traj_encoder", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="Toggles whether or not to use Transformer version of meta-controller.")
    parser.add_argument("--use_tf_seq_obs_encoder", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="Toggles whether or not to use Transformer version of meta-controller.")
    parser.add_argument("--use_pc_extractor", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggles whether or not to use Transformer version of meta-controller.")
    parser.add_argument("--total_timesteps", type=int, default=int(1e9), help="total timesteps of the experiments")
    parser.add_argument("--num_envs", type=int, default=10, help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=80, help="the number of steps to run in each environment per policy rollout per object")
    parser.add_argument("--pc_batchsize", type=int, default=None, help="the number of steps to run in each environment per policy rollout per object")
    parser.add_argument("--use_relu", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Use Relu or tanh.")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Use GAE for advantage computation")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="the lambda for the general advantage estimation")
    parser.add_argument("--minibatch-size", type=int, default=1000, help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10, help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=2., help="the target KL divergence threshold")
    parser.add_argument("--deterministic", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="Using deterministic policy instead of normal")
    parser.add_argument('--eval', type=bool, default=False, help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor for reward (default: 0.95)')
    parser.add_argument('--tau', type=float, default=0.0005, metavar='G', help='target smoothing coefficient(τ) (default: 0.0005)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='G', help='learning rate (default: 0.00001)')  # first 0.0001 then 0.00005
    parser.add_argument('--alpha', type=float, default=0.05, metavar='G', help='Temperature parameter α determines the relative importance of the entropy\
                                    term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G', help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N', help='random seed (default: 123456)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N', help='hidden size (default: 256)')
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="if toggled, cuda will be enabled by default")
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--index_episode', type=str, default='best')
    parser.add_argument('--random_policy', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--sequence_len', type=int, default=5)
    parser.add_argument('--reward_episodes', type=int, default=5000)
    parser.add_argument('--cpus', type=int, default=[], nargs='+', help="run environments on specified cpus")
    parser.add_argument("--torch_deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")

    args = parser.parse_args()

    # Training required attributes
    args.step_sync = True
    args.pc_batchsize = args.pc_batchsize if args.pc_batchsize is not None else args.num_envs

    if args.cpus:
        print('Running on specific CPUS:', args.cpus)
        process = psutil.Process()
        process.cpu_affinity(args.cpus)

    if args.realtime:
        args.rendering = True

    # Uniformalize training name
    additional = 'Sync'
    ###--- suffix for final name ---###
    if args.specific_scene is not None:
        additional += f'_{args.specific_scene}'
    if args.use_traj_encoder: 
        additional += '_TrajEncoderTF' if args.use_tf_traj_encoder else '_TrajEncoderFC'
    if args.use_seq_obs_encoder:
        additional += '_SeqObsEncoderTF' if args.use_tf_seq_obs_encoder else '_SeqObsEncoderFC'
    if args.use_pc_extractor: 
        additional += '_PCExtractor'
    if args.checkpoint is not None: additional += '_FineTune'
    if args.use_relu: additional += '_Relu'
    else: additional += '_Tanh'
    additional += '_Rand'
    if args.random_select_objs_pool: additional += '_ObjPool'
    if args.random_select_placing: additional += '_ObjPlace'
    if args.random_select_scene_pool: additional += '_ScenePool'
    if not args.fixed_scene_only: additional += '_unFixedScene'
    if not args.fixed_qr_region: additional += '_QRRegion'
    additional += '_Goal'
    if args.use_curriculum: additional += '_Curriculum'
    if args.min_num_placing_objs: additional += f'_minObjNum{args.min_num_placing_objs}'
    if args.train_step: additional += f'_objStep{args.train_step}'
    if args.max_num_placing_objs: additional += f'_maxObjNum{args.max_num_placing_objs}'
    if args.num_pool_objs: additional += f'_maxPool{args.num_pool_objs}'
    if args.num_pool_scenes: additional += f'_maxScene{args.num_pool_scenes}'
    if args.max_stable_steps: additional += f'_maxStable{args.max_stable_steps}'
    if args.min_continue_stable_steps: additional += f'_contStable{args.min_continue_stable_steps}'
    if args.num_episode_to_replace_pool: additional += f'_Epis2Replace{args.num_episode_to_replace_pool}'
    additional += '_Weight'
    if args.reward_pobj > 0: additional += f'_rewardPobj{args.reward_pobj}'
    if args.penalty > 0: additional += f'_ori{args.penalty}'
    additional += f'_seq{args.sequence_len}'
    additional += f'_step{args.num_steps}'
    additional += f'_trial{args.max_trials}'

    args.timer = '_' + '_'.join(str(datetime.datetime.now())[5:16].split())  # a time name file

    if args.random_policy:  # final_name is in all file names: .csv / .json / trajectory / checkpoints
        args.final_name = args.object_pool_name + args.timer + additional.replace('-train', '-random_policy')
    elif args.force_name:
        args.final_name = args.force_name + args.timer
    else: # Normal training
        args.final_name = args.object_pool_name + args.timer + additional  # only use final name
    print(f"Uniform Name: {args.final_name}")

    ###### Saving Results ######
    # create result folder
    args.result_dir = os.path.join(args.result_dir, args.object_pool_name)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # create csv folder
    args.csv_dir = os.path.join(args.result_dir, 'CSV')
    if args.collect_data and not os.path.exists(args.csv_dir):
        os.makedirs(args.csv_dir)
    args.result_file_path = os.path.join(args.csv_dir, args.final_name + '.csv')

    # create checkpoints folder; not create if use expert action
    args.checkpoint_dir = os.path.join(args.result_dir, 'checkpoints', args.final_name)
    if args.collect_data and not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # create trajectory folder
    args.trajectory_dir = os.path.join(args.result_dir, 'trajectories', args.final_name)
    if args.collect_data and not os.path.exists(args.trajectory_dir):
        os.makedirs(args.trajectory_dir)

    # create json folder
    args.json_dir = os.path.join(args.result_dir, 'Json')
    if args.collect_data and not os.path.exists(args.json_dir):
        os.makedirs(args.json_dir)
    args.json_file = os.path.join(args.json_dir, args.final_name + '.json')

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.collect_data:
        with open(args.json_file, 'w') as json_obj:
            json.dump(vars(args), json_obj, indent=4)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env and scene setup
    if args.num_envs > 1:
        envs = create_multi_envs(args, 'forkserver')
        temp_env = envs.tempENV; tensor_dtype = temp_env.tensor_dtype
    elif args.num_envs == 1:
        envs = RoboSensaiBullet(args=args)
        temp_env = envs; tensor_dtype = temp_env.tensor_dtype
    
    agent = Agent(temp_env).to(device)
    if args.checkpoint is not None:
        checkpoint_folder = os.path.join(args.result_dir, 'checkpoints', args.checkpoint)
        args.checkpoint_path = os.path.join(checkpoint_folder, args.checkpoint + '_' + args.index_episode)
        assert os.path.exists(args.checkpoint_path), f"Checkpoint path {args.checkpoint_path} does not exist!"
        agent.load_checkpoint(args.checkpoint_path, map_location=device)
    # agent = torch.compile(agent) # Speed up the model
    agent.set_mode('train')  # set to train
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    # Assuming you have these variables from your environment
    raw_obs_shape_data = [
        ["Raw Observation Shape", ""],
        ["Name", "Shape"],
        ["Raw Observation Shape", temp_env.raw_act_hist_qr_obs_shape],
        ["QR Region Dim", temp_env.qr_region_dim],
        ["Action Dim", temp_env.action_dim],
        ["Traj History Dim", temp_env.traj_hist_dim],
        ["Scene PC Dim", f"(3, {args.max_num_scene_points})"],
        ["Obj PC Dim", f"(3, {args.max_num_urdf_points})"],
        ["Sequence Length", f"{args.sequence_len}"]
    ]
    
    post_obs_shape_data = [
        ["Post Observation Shape", ""],
        ["Name", "Shape"],
        ["Post Observation Shape", temp_env.post_observation_shape],
        ["Scene Feature Dim", temp_env.scene_ft_dim],
        ["Obj Feature Dim", temp_env.obj_ft_dim],
        ["Seq Obs ft Dim", temp_env.seq_info_ft_dim]
    ]

    print(tabulate(raw_obs_shape_data, headers="firstrow", tablefmt="grid"))
    print(tabulate(post_obs_shape_data, headers="firstrow", tablefmt="grid"))

    # wandb
    config = dict(
        Name=args.env_name,
        algorithm='PPO Continuous',
        num_envs=args.num_envs,
        max_traj_len=args.max_traj_history_len,
        lr=args.lr,
        gamma=args.gamma,
        alpha=args.ent_coef,
        deterministic=args.deterministic,
        sequence_len=args.sequence_len,
        random_policy=args.random_policy,
    )

    name = args.final_name
    if args.collect_data and args.wandb:
        wandb.init(project=args.env_name, entity='jiayinsen', config=config, name=name)
    else:
        wandb.init(mode="disabled")

    # ALGO Logic: Storage setup
    print(f"Observation Shape: {temp_env.post_observation_shape}")
    if args.use_curriculum:
        num_placing_objs_lst = list(range(args.min_num_placing_objs, args.max_num_placing_objs, args.train_step))
        if args.max_num_placing_objs not in num_placing_objs_lst: num_placing_objs_lst.append(args.max_num_placing_objs)
    else:
        num_placing_objs_lst = [args.max_num_placing_objs]

    # Global variables that will not be reset
    global_update_iter = 0
    global_step = 0
    global_episodes = 0
    # Record episodes, iters, and steps are used to fill the reward buffer
    reward_update_iters = 0
    reward_steps = 0
    reward_episodes = 0

    save_mile_stone = 0
    meta_data = {"milestone": {}, "training_info": {}}
    best_agent_state_dict, best_optimizer_state_dict = None, None

    for num_placing_objs in num_placing_objs_lst:
        if args.num_envs > 1:
            envs.env_method('set_args', 'max_num_placing_objs', num_placing_objs)
            envs.env_method('reset_info') # reset the info to record the new training miscs
        else:
            envs.args.max_num_placing_objs = num_placing_objs
            envs.reset_info()
        
        args.num_steps = max(args.num_steps, ((args.max_trials * num_placing_objs) * 4)) # At least 4 * num_envs or 80/avg_steps global_episodes to update
        args.batch_size = int(args.num_envs * args.num_steps)
        args.num_minibatches = ceil(args.batch_size // args.minibatch_size)
        meta_data["training_info"].update({
            num_placing_objs: {
                "batch_size": args.batch_size,
                "num_steps": args.num_steps,
            }
        })

        # Load the best parameters for next curriculum
        if best_agent_state_dict is not None:
            agent.load_state_dict(best_agent_state_dict)
            best_agent_state_dict = None # Consume the best agent state dict
        if best_optimizer_state_dict is not None:
            optimizer.load_state_dict(best_optimizer_state_dict)
            best_optimizer_state_dict = None # Consume the best optimizer state dict

        # Storage
        seq_obs = torch.zeros((args.num_steps, args.num_envs) + temp_env.raw_act_hist_qr_obs_shape[1:], dtype=tensor_dtype).to(device)
        scene_ft_obs = torch.zeros((args.num_steps, args.num_envs) + (temp_env.scene_ft_dim, ), dtype=tensor_dtype).to(device)
        obj_ft_obs = torch.zeros((args.num_steps, args.num_envs) + (temp_env.obj_ft_dim, ), dtype=tensor_dtype).to(device)
        actions = torch.zeros((args.num_steps, args.num_envs) + temp_env.action_shape[1:], dtype=temp_env.tensor_dtype).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=temp_env.tensor_dtype).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs), dtype=temp_env.tensor_dtype).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs), dtype=temp_env.tensor_dtype).to(device)
        values = torch.zeros((args.num_steps, args.num_envs), dtype=temp_env.tensor_dtype).to(device)

        # TRY NOT TO MODIFY: start the game
        start_time = time.time()
        next_seq_obs = torch.Tensor(envs.reset()).to(device)
        # Scene and obj feature tensor are keeping updated inplace
        next_scene_ft_obs = torch.zeros((args.num_envs, ) + (temp_env.scene_ft_dim, ), dtype=tensor_dtype).to(device)
        next_obj_ft_obs = torch.zeros((args.num_envs, ) + (temp_env.obj_ft_dim, ), dtype=tensor_dtype).to(device)
        reset_infos = envs.reset_infos if args.num_envs > 1 else [envs.info]
        agent.preprocess_pc_update_tensor(next_scene_ft_obs, next_obj_ft_obs, reset_infos, use_mask=True)
        next_done = torch.zeros(args.num_envs).to(device)
        num_updates = args.total_timesteps // args.batch_size  # ?? same as global_episodes? No!! global_episodes = (total_timsteps / batch_size) * num_envs * (avg num_episodes in 128 steps, usually are 20)

        # custom record information
        episode_rewards = torch.zeros((args.num_envs, ), dtype=temp_env.tensor_dtype).to(device)
        episode_pos_rewards = torch.zeros((args.num_envs, ), dtype=temp_env.tensor_dtype).to(device)
        episode_ori_rewards = torch.zeros((args.num_envs, ), dtype=temp_env.tensor_dtype).to(device)
        episode_act_penalties = torch.zeros((args.num_envs, ), dtype=temp_env.tensor_dtype).to(device)

        episode_rewards_box = torch.zeros((args.reward_episodes, ), dtype=temp_env.tensor_dtype).to(device)
        episode_success_box = torch.zeros((args.reward_episodes, ), dtype=temp_env.tensor_dtype).to(device)
        episode_placed_objs_box = torch.zeros((args.reward_episodes, ), dtype=tensor_dtype).to(device)
        iter_success_rate_box = torch.zeros((args.patience_iters, ), dtype=temp_env.tensor_dtype).to(device)

        pos_r_box = torch.zeros((args.reward_episodes, ), dtype=temp_env.tensor_dtype).to(device)
        ori_r_box = torch.zeros((args.reward_episodes, ), dtype=temp_env.tensor_dtype).to(device)
        act_p_box = torch.zeros((args.reward_episodes, ), dtype=temp_env.tensor_dtype).to(device)
        best_acc = 0; curri_episodes = 0; curri_steps = 0; curri_update_iters = 0; avg_buffer_reset = True

        # training
        for update in range(1, num_updates + 1):

            # torch.cuda.empty_cache()

            # Annealing the rate if instructed to do so.
            if args.anneal_lr:  # schedule learning rate
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * args.lr
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                global_step += args.num_envs
                curri_steps += args.num_envs
                seq_obs[step] = next_seq_obs
                scene_ft_obs[step] = next_scene_ft_obs
                obj_ft_obs[step] = next_obj_ft_obs

                dones[step] = next_done

                ## ----- ALGO LOGIC: action logic ----- ##
                ## if not expert_action, normal training; Otherwise use only expert actions
                # transfer discrete actions to real actions; TODO: Logical problem about next_seq_obs (terminal observation to query step action for the first action)
                if args.random_policy:
                    step_action = torch.rand((args.num_envs, temp_env.action_shape[1]), device=device)
                else:
                    with torch.no_grad():
                        step_action, logprob, _, value = agent.get_action_and_value([next_seq_obs, next_scene_ft_obs, next_obj_ft_obs])
                        values[step] = value.flatten()
                    actions[step] = step_action
                    logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_seq_obs, reward, done, infos = envs.step(step_action)
                update_env_ids = agent.preprocess_pc_update_tensor(next_scene_ft_obs, next_obj_ft_obs, infos, use_mask=True)

                next_seq_obs, next_done = torch.Tensor(next_seq_obs).to(device), torch.Tensor(done).to(device)
                rewards[step] = torch.Tensor(reward).to(device).view(-1) # if reward is not tensor inside
                episode_rewards += rewards[step]
                
                terminal_index = next_done == 1
                terminal_nums = terminal_index.sum().item()
                # Compute the average episode rewards.
                if terminal_nums > 0:
                    global_episodes += terminal_nums
                    curri_episodes += terminal_nums
                    update_tensor_buffer(episode_rewards_box, episode_rewards[terminal_index])
                    update_tensor_buffer(pos_r_box, episode_pos_rewards[terminal_index])
                    update_tensor_buffer(act_p_box, episode_act_penalties[terminal_index])
                    
                    terminal_ids = terminal_index.nonzero().flatten()
                    success_buf = torch.Tensor(combine_envs_float_info2list(infos, 'success', terminal_ids)).to(device)
                    update_tensor_buffer(episode_success_box, success_buf)
                    placed_obj_num_buf = torch.Tensor(combine_envs_float_info2list(infos, 'success_placed_obj_num', terminal_ids)).to(device)
                    update_tensor_buffer(episode_placed_objs_box, placed_obj_num_buf)

                    # Empty the episode rewards
                    episode_rewards[terminal_index] = 0.
                    episode_reward = torch.mean(episode_rewards_box[-curri_episodes:]).item()
                    episode_success_rate = torch.mean(episode_success_box[-curri_episodes:]).item()
                    episode_placed_objs = torch.mean(episode_placed_objs_box[-curri_episodes:]).item()

                    if not args.quiet:
                        print(f"Global Steps:{global_step}/{args.total_timesteps}," 
                              f"Episode:{global_episodes}, Success Rate:{episode_success_rate:.2f},"
                              f"Reward:{episode_reward:.4f}")
                    
                    if args.collect_data:
                        # Save success rate and placed objects number
                        meta_data['training_info'].update({
                            num_placing_objs: {
                                "global_episodes": global_episodes,
                                "global_steps": global_step,
                                "scene_obj_success_num": combine_envs_dict_info2dict(infos, key="scene_obj_success_num"),
                                "obj_success_rate": combine_envs_dict_info2dict(infos, key="obj_success_rate"),
                            }
                        })

                        if args.wandb:
                            wandb.log({'global_episodes': global_episodes, 
                                       'global_steps': global_step,
                                       'global_iterations': global_update_iter,
                                       'reward/reward_train': episode_reward})

                            if curri_episodes >= args.reward_episodes:  # episode success rate
                                if avg_buffer_reset: # Only add once per curriculum
                                    reward_episodes += curri_episodes
                                    reward_update_iters += curri_update_iters
                                    reward_steps += curri_steps
                                    avg_buffer_reset = False
                                wandb.log({'s_episodes': global_episodes - reward_episodes, 
                                           's_iterations': global_update_iter - reward_update_iters,
                                           's_steps': global_step - reward_steps,
                                           'reward/success_rate': episode_success_rate,
                                           'reward/num_placed_objs': episode_placed_objs})

                        if curri_episodes > args.reward_episodes and episode_success_rate > best_acc:
                            best_acc = episode_success_rate
                            best_agent_state_dict = deepcopy(agent.state_dict())
                            best_optimizer_state_dict = deepcopy(optimizer.state_dict())
                            agent.save_checkpoint(folder_path=args.checkpoint_dir,
                                                folder_name=args.final_name, 
                                                suffix='best')
                            print(f"s_episodes: {global_episodes - reward_episodes} | " 
                                  f"Now best accuracy is {best_acc * 100:.3f}% | " 
                                  f"Number of placed objects is {episode_placed_objs:.2f}")
                        
                        # about every args.reward_episodes (because of multi-envs) global_episodes to save one model
                        if (global_episodes - save_mile_stone) >= args.reward_episodes:
                            agent.save_checkpoint(folder_path=args.checkpoint_dir, 
                                                  folder_name=args.final_name,
                                                  suffix=str(global_episodes))
                            save_mile_stone = global_episodes
                            save_json(meta_data, os.path.join(args.trajectory_dir, "meta_data.json"))


            ####----- force action to test variance; Skip the training process ----####
            if args.random_policy: continue

            ####----- Curriculum next stage checker ----####
            update_tensor_buffer(iter_success_rate_box, torch.tensor([episode_success_rate], dtype=iter_success_rate_box.dtype, device=iter_success_rate_box.device))
            if args.use_curriculum and \
                num_placing_objs < args.max_num_placing_objs and \
                curri_update_iters >= args.patience_iters:
                if iter_success_rate_box.max()>=0.8 or \
                    iter_success_rate_box.max()-iter_success_rate_box.min()<=0.01:
                    if args.collect_data: 
                        meta_data["milestone"].update({
                                num_placing_objs: {
                                "num_placing_objs": num_placing_objs,
                                "success_rate": best_acc,
                                's_episodes': global_episodes - reward_episodes, 
                                's_iterations': global_update_iter - reward_update_iters,
                                's_steps': global_step - reward_steps,
                            }
                        })
                        save_json(meta_data, os.path.join(args.trajectory_dir, "meta_data.json"))
                    print(f"Curriculum update: {num_placing_objs} -> {num_placing_objs + args.train_step}")
                    break


            ####----- Compute advantage for each state in the markov chain ----####
            with torch.no_grad():
                next_value = agent.get_value([next_seq_obs, next_scene_ft_obs, next_obj_ft_obs]).reshape(1, -1)
                if args.gae:
                    advantages = torch.zeros_like(rewards).to(device)
                    lastgaelam = 0
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                        advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    returns = advantages + values
                else:
                    returns = torch.zeros_like(rewards).to(device)
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            next_return = returns[t + 1]
                        returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                    advantages = returns - values

            # flatten the batch
            b_seq_obs = seq_obs.reshape((-1,) + temp_env.raw_act_hist_qr_obs_shape[1:])
            b_scene_ft_obs = scene_ft_obs.reshape((-1,) + (temp_env.scene_ft_dim, ))
            b_obj_ft_obs = obj_ft_obs.reshape((-1,) + (temp_env.obj_ft_dim, ))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + temp_env.action_shape[1:])
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            # Save previous parameters
            if args.target_kl is not None:
                agent_params_store = copy.deepcopy(agent.state_dict())
                optim_params_store = copy.deepcopy(optimizer.state_dict())

            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value([b_seq_obs[mb_inds], b_scene_ft_obs[mb_inds], b_obj_ft_obs[mb_inds]], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                        agent.load_state_dict(agent_params_store)
                        optimizer.load_state_dict(optim_params_store)
                        break

            if args.target_kl is not None and approx_kl > args.target_kl:
                continue # Skip the wandb log since the update is not successful

            global_update_iter += 1
            curri_update_iters += 1
            # To float32 is because it does support for bfloat16 to numpy
            y_pred, y_true = b_values.to(torch.float32).cpu().numpy(), b_returns.to(torch.float32).cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            if args.collect_data and args.wandb:
                wandb.log({
                    'steps': global_step, 
                    'iterations': global_update_iter,
                    'train/learning_rate': optimizer.param_groups[0]["lr"],
                    'train/critic_loss': v_loss.item(),
                    'train/policy_loss': pg_loss.item(),
                    'train/entropy': entropy_loss.item(),
                    'train/approx_kl': approx_kl.item(),
                    'train/advantages': mb_advantages.mean().item(),
                    'train/explained_variance': explained_var
                })

            if not args.quiet:
                print("Running Time:", convert_time(time.time() - start_time), "Global Steps", global_step)


    if args.collect_data and not args.random_policy:  # not random policy or expert action
        agent.save_checkpoint(folder_path=args.checkpoint_dir, folder_name=args.final_name, suffix='last')  # last
    print('Process Over here')
    envs.close()
    wandb.finish()