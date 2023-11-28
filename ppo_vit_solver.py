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

from isaacgym.gymutil import parse_device_str
from RoboSensai_env import *
from PPO.PPO_discrete_vit import *

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Train Tactile Pushing Experiment')
    
    # Env hyper parameters
    parser.add_argument('--collect_data', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True) # https://docs.python.org/3/library/argparse.html#:~:text=%27%3F%27.%20One%20argument,to%20illustrate%20this%3A
    parser.add_argument('--object_name', type=str, default='cube', help="Target object to be grasped. Ex: cube")
    parser.add_argument('--robot_config_name', type=str, default='ur5_robotiq', help="name of robot configs to load from grasputils. Ex: ur5_robotiq")
    parser.add_argument('--rendering', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--realtime', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--num_trials', type=int, default=100)  # database length if have
    parser.add_argument('--planner', type=str, default='birrt')
    parser.add_argument('--path_visualize', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--quiet', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)

    # Isaac Gym parameters
    parser.add_argument('--headless', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Run headless without creating a viewer window')
    parser.add_argument('--nographics', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Disable graphics context creation, no viewer window is created, and no headless rendering is available')
    parser.add_argument('--use_gpu_pipeline', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help='Disable graphics context creation, no viewer window is created, and no headless rendering is available')
    parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
    parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics Device ID')
    parser.add_argument('--num_threads', type=int, default=0, help='Number of cores used by PhysX')
    parser.add_argument('--subscenes', type=int, default=0, help='Number of PhysX subscenes to simulate in parallel')
    parser.add_argument('--slices', type=int, help='Number of client threads that process env slices')

    # Handem Env parameters
    parser.add_argument('--task', type=str, default="P", help='Pushing Task Training')
    parser.add_argument('--random_target_init', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help='Randomize goal pose')
    parser.add_argument('--random_goal', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help='Randomize goal pose')
    parser.add_argument('--random_target', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Randomize target pose')
    parser.add_argument('--include_target_obs', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Whether target observation is included in the states')
    parser.add_argument('--include_finger_vel', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Include finger discretized velocity into observation')
    parser.add_argument('--use_world_frame_obs', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='The observation space in finger frame or world frame')
    parser.add_argument('--use_abstract_contact_obs', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Use human abstract observation')
    parser.add_argument('--use_contact_torque', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Use contact torque instead of contact location (default)')
    parser.add_argument('--use_2D_contact', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Use 2D contact observation')
    parser.add_argument('--use_contact_force', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Use contact torque instead of contact location (default)')
    parser.add_argument('--filter_contact', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Mask out half of finger contact')
    parser.add_argument('--constrain_ws', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help='Restrict the finger movement within a certain space')
    parser.add_argument('--add_random_noise', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Add random noise to contact info')
    parser.add_argument('--use_relative_goal', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Use relative goal position or global goal position as the target goal')
    parser.add_argument('--contact_noise_v', type=float, default=0.01, help='Contact position noise range')
    parser.add_argument('--force_noise_v', type=float, default=0.0, help='Contact force noise range')
    parser.add_argument('--pos_weight', type=float, default=40., help='Position reward weight')
    parser.add_argument('--ori_weight', type=float, default=0., help='Orientation reward weight')
    parser.add_argument('--act_weight', type=float, default=0., help='Action penalty weight')
    parser.add_argument('--draw_contact', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='Draw contact force direction')

    # I/O hyper parameter
    parser.add_argument('--assets_folder', type=str, default='assets', help="folder path that stores all urdf files")
    parser.add_argument('--debug', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--result_dir', type=str, default='train_res', required=False)
    parser.add_argument('--baseline_experiment_path', type=str, default='a_new_assets', help='use motion path in this file for the run')  # 'a_new_assets'
    parser.add_argument('--wandb', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--force_name', default=None, type=str)

    # Algorithm specific arguments
    parser.add_argument('--env_name', default="RoboSensai_Solver", help='Pybullet environment')
    parser.add_argument("--teacher_critic", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="Include target observation in critic obs states but not actor states.")
    parser.add_argument("--use_lstm", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="Toggles whether or not to use LSTM version of meta-controller.")
    parser.add_argument("--use_transformer", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggles whether or not to use Transformer version of meta-controller.")
    parser.add_argument("--total_timesteps", type=int, default=int(1e9), help="total timesteps of the experiments")
    parser.add_argument("--num_envs", type=int, default=10, help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=32, help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Use GAE for advantage computation")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4, help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=8, help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None, help="the target KL divergence threshold")
    parser.add_argument('--policy', default="Gaussian", help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
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
    parser.add_argument('--load_checkpoint', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--random_policy', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--expert_action', type=float, default=None, metavar='N', nargs='*', help='expert action run to see the variance')
    parser.add_argument('--sequence_len', type=int, default=10)
    parser.add_argument('--assigned_reward', type=int, default=1) 
    parser.add_argument('--reward_steps', type=int, default=10000)
    parser.add_argument('--cpus', type=int, default=[], nargs='+', help="run environments on specified cpus")
    parser.add_argument("--torch_deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")

    args = parser.parse_args()

    # Isaac Gym operations
    args.sim_device_type, args.compute_device_id = parse_device_str(args.sim_device)
    args.use_gpu = (args.sim_device_type == 'cuda')
    # Using --nographics implies --headless
    if args.nographics: args.headless = True
    if args.slices is None: args.slices = args.subscenes

    # Training required attributes
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    if args.cpus:
        print('Running on specific CPUS:', args.cpus)
        process = psutil.Process()
        process.cpu_affinity(args.cpus)

    if args.realtime:
        args.rendering = True

    # Training flags check
    if args.use_gpu_pipeline:
        assert args.use_contact_torque is False, "Error: use_contact_torque only supports in CPU pipeline!"
    else:
        assert args.use_abstract_contact_obs is False, "Error: use_abstract_contact_obs only supports in GPU pipeline"

    # Uniformalize training name
    additional = ''; additional += '_Pushing'
    ###--- suffix for final name ---###
    if args.use_lstm: additional += '_LSTM'
    elif args.use_transformer: additional += '_Transformer'
    else: additional += '_FC'
    if args.use_gpu_pipeline: additional += '_GPU'
    else: additional += '_CPU'
    additional += '_Rand'
    if args.random_goal: additional += '_goal'
    if args.random_target: additional += '_target'
    if args.random_target_init: additional += '_targInit'
    additional += '_With'
    if args.include_target_obs: additional += '_targetObs'
    if args.include_finger_vel: additional += '_finVel'
    if args.use_contact_force: additional += '_force'
    additional += '_Use'
    if args.use_world_frame_obs: additional += '_worldObs'
    if args.use_contact_torque: additional += '_torqObs'
    if args.use_abstract_contact_obs: additional += '_abstObs'
    if args.use_2D_contact: additional += '_2dObs'
    additional += '_Add'
    if args.add_random_noise: additional += '_noise'
    if args.filter_contact: additional += '_filter'
    if args.constrain_ws: additional += '_limitws'
    additional += '_Weight'
    if args.pos_weight > 0: additional += f'_pos{args.pos_weight}'
    if args.ori_weight > 0: additional += f'_ori{args.ori_weight}'
    if args.act_weight > 0: additional += f'_actP{args.act_weight}'

    args.timer = '_' + '_'.join(str(datetime.datetime.now())[5:16].split())  # a time name file

    if args.random_policy:  # final_name is in all file names: .csv / .json / trajectory / checkpoints
        args.final_name = args.object_name + args.timer + additional.replace('-train', '-random_policy')
    elif args.force_name:
        args.final_name = args.force_name + args.timer
    elif args.load_checkpoint: # Resume training
        args.final_name = args.checkpoint
    else: # Normal training
        args.final_name = args.object_name + args.timer + additional  # only use final name
    print(f"Uniform Name: {args.final_name}")

    ###### Saving Results ######
    # create result folder
    args.result_dir = os.path.join(args.result_dir, args.object_name)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # create csv folder
    args.csv_dir = os.path.join(args.result_dir, 'CSV')
    if args.collect_data and not os.path.exists(args.csv_dir):
        os.makedirs(args.csv_dir)
    args.result_file_path = os.path.join(args.csv_dir, args.final_name + '.csv')

    # create checkpoints folder; not create if use expert action
    args.checkpoint_dir = os.path.join(args.result_dir, 'checkpoints', args.final_name)
    if args.collect_data and not args.expert_action and not os.path.exists(args.checkpoint_dir):
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

    # env and scene setup; TODO Input the aruments into HandemEnv
    envs = RoboSensaiEnv(args=args)
    agent = Agent(envs).to(device)
    agent.set_mode('train')  # set to train
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    # ALGO Logic: Storage setup
    print(f"Image observation Shape: {envs.single_img_obs_dim}\n",
          f"Proprioception observation Shape: {envs.single_proprioception_dim}\n",
          f"Action Shape: {envs.action_dim}\n",
          f"Agent input size: {agent.agent_input_size}\n")
    
    vis_obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_img_obs_dim[1:]).to(device)
    vec_obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_proprioception_dim[1:]).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.action_dim[1:]).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs_dict = envs.reset()
    next_vis_obs, next_vec_obs = next_obs_dict['image'].to(device), next_obs_dict['proprioception'].to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size  # ?? same as episodes? No!! episodes = (total_timsteps / batch_size) * num_envs * (avg num_episodes in 128 steps, usually are 20)

    # wandb
    config = dict(
        Name=args.env_name,
        algorithm='PPO Discrete',
        num_updates=num_updates,
        lr=args.lr,
        gamma=args.gamma,
        alpha=args.ent_coef,
        batch_size=args.batch_size,
        policy=args.policy,
        sequence_len=args.sequence_len,
        assigned_reward=args.assigned_reward,
        random_policy=args.random_policy,
        expert_action=args.expert_action,
    )
    resume = True if args.load_checkpoint else False
    name = args.final_name
    if args.collect_data and args.wandb:
        wandb.init(project=args.env_name, entity='RoboSensai', config=config, resume=resume, name=name)
    else:
        wandb.init(mode="disabled")

    # custom record information
    episode_rewards = torch.zeros((args.num_envs, )).to(envs.device)
    episode_pos_rewards = torch.zeros((args.num_envs, )).to(envs.device)
    episode_ori_rewards = torch.zeros((args.num_envs, )).to(envs.device)
    episode_act_penalties = torch.zeros((args.num_envs, )).to(envs.device)

    episode_rewards_box = torch.zeros((args.reward_steps, )).to(envs.device)
    episode_success_box = torch.zeros((args.reward_steps, )).to(envs.device)
    pos_r_box = torch.zeros((args.reward_steps, )).to(envs.device)
    ori_r_box = torch.zeros((args.reward_steps, )).to(envs.device)
    act_p_box = torch.zeros((args.reward_steps, )).to(envs.device)
    best_acc = 0; i_episode = 0; mile_stone = 0

    # training
    for update in range(1, num_updates + 1):

        # torch.cuda.empty_cache()

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:  # schedule learning rate
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.lr
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            vis_obs[step], vec_obs[step] = next_vis_obs, next_vec_obs
            dones[step] = next_done

            ## ----- ALGO LOGIC: action logic ----- ##
            ## if not expert_action, normal training; Otherwise use only expert actions
            # transfer discrete actions to real actions; TODO: Logical problem about next_obs (terminal observation to query step action for the first action)
            if not args.expert_action and not args.random_policy:
                with torch.no_grad():
                    step_action, logprob, _, value = agent.get_action_and_value((next_vis_obs, next_vec_obs))
                    values[step] = value.flatten()
                actions[step] = step_action
                logprobs[step] = logprob
            elif args.expert_action:
                step_action = [args.expert_action] * args.num_envs
            elif args.random_policy:
                step_action = envs.random_actions()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs_dict, reward, done, infos = envs.step(step_action)
            next_vis_obs, next_vec_obs = next_obs_dict['image'], next_obs_dict['proprioception']
            next_vis_obs = torch.Tensor(next_vis_obs).to(device)
            next_vec_obs = torch.Tensor(next_vec_obs).to(device)
            # rewards[step] = torch.tensor(reward).to(device).view(-1) # if reward is not tensor inside
            rewards[step] = reward.to(device).view(-1)
            next_done = torch.Tensor(done).to(device)

            #####################################################################
            ###==================Record all rewards information===============###
            #####################################################################
            pos_reward, act_penalty = infos['pos_reward'], infos['act_penalty']

            episode_rewards += reward
            episode_pos_rewards += pos_reward
            episode_act_penalties += act_penalty
            
            terminal_index = done == 1
            terminal_nums = terminal_index.sum().item()
            # Compute the average episode rewards.
            if terminal_nums > 0:
                i_episode += terminal_nums

                update_tensor_buffer(episode_rewards_box, episode_rewards[terminal_index])
                update_tensor_buffer(pos_r_box, episode_pos_rewards[terminal_index])
                update_tensor_buffer(act_p_box, episode_act_penalties[terminal_index])
                update_tensor_buffer(episode_success_box, envs.success_buf[terminal_index])

                episode_rewards[terminal_index] = 0.
                episode_pos_rewards[terminal_index] = 0.
                episode_act_penalties[terminal_index] = 0.

                episode_reward = torch.mean(episode_rewards_box).item()
                episode_pos_r = torch.mean(pos_r_box).item()
                episode_act_p = torch.mean(act_p_box).item()
                
                episode_success_rate = torch.mean(episode_success_box).item()

                if not args.quiet:
                    print(f"Global Steps:{global_step}/{args.total_timesteps}, Episode:{i_episode}, Success Rate:{episode_success_rate:.2f}, Reward:{episode_reward:.4f}," \
                        f"Pos Reward: {episode_pos_r:.4f}, Act Penalty: {episode_act_p:.4f}")
                
                if args.collect_data:
                    if args.wandb:
                        wandb.log({'episodes': i_episode, 'reward/reward_train': episode_reward, 'reward/reward_pos': episode_pos_r,
                                   'reward/penalty_act': episode_act_p})

                        if i_episode >= args.reward_steps:  # episode success rate
                            wandb.log({'s_episodes': i_episode - args.reward_steps, 'reward/success_rate': episode_success_rate})

                    if not args.expert_action:  # if expert action, it is not considered to save result
                        if episode_success_rate > best_acc and i_episode > args.reward_steps:  # at least after 500 episodes could consider as a good success
                            best_acc = episode_success_rate;
                            agent.save_checkpoint(folder_path=args.checkpoint_dir,
                                                  folder_name=args.final_name, suffix='best')
                            print(f'Now best accuracy is {best_acc * 100}%')
                        if (i_episode - mile_stone) >= args.reward_steps:  # about every args.reward_steps episodes to save one model
                            agent.save_checkpoint(folder_path=args.checkpoint_dir, folder_name=args.final_name,
                                                  suffix=str(i_episode))
                            mile_stone = i_episode


        ####----- force action to test variance; Skip the training process ----####
        if args.expert_action: continue
        if args.random_policy: continue

        ####----- Compute advantage for each state in the markov chain ----####
        with torch.no_grad():
            next_value = agent.get_value((next_vis_obs, next_vec_obs)).reshape(1, -1)
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
        b_vis_obs = vis_obs.reshape((-1,) + envs.single_img_obs_dim[1:])
        b_vec_obs = vec_obs.reshape((-1,) + envs.single_proprioception_dim[1:])
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.action_dim[1:])
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)

        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value((b_vis_obs[mb_inds], b_vec_obs[mb_inds]), b_actions[mb_inds].T)  # batch actions need to transpose for computation because it is easy for "zip" operation
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

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if args.collect_data and args.wandb:
            wandb.log({'steps': global_step, 
                       'train/learning_rate': optimizer.param_groups[0]["lr"],
                       'train/critic_loss': v_loss.item(),
                       'train/policy_loss': pg_loss.item(),
                       'train/entropy': entropy_loss.item(),
                       'train/approx_kl': approx_kl.item(),
                       'train/explained_variance': explained_var})

        if not args.quiet:
            print("Running Time:", convert_time(time.time() - start_time), "Global Steps", global_step)


    if args.collect_data and not args.expert_action:  # not expert action
        agent.save_checkpoint(folder_path=args.checkpoint_dir, folder_name=args.final_name, suffix='last')  # last
    print('Process Over here')
    envs.close()
    wandb.finish()