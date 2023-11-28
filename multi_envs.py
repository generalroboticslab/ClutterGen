import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import PPO
from RoboSensai_bullet import RoboSensaiBullet
import copy


class CustomPyBulletEnv(gym.Env):
    def __init__(self, args):
        # Initialize your custom PyBullet env here
        self.env = RoboSensaiBullet(args)
        
        # Define your action and observation spaces (FC)
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.env.action_shape[1],), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(np.prod(self.env.observation_shape[1:]), ), dtype=np.float32)


    def reset(self, seed=None):
        # Reset the env and return the initial observation
        observation = self.env.reset()
        info = self.env.info
        return observation, info


    def step(self, action):
        # Perform a step in the env based on the given action
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, False, info


    def render(self, mode='human'):
        # Render the env (optional)
        self.env.render(mode)


    def close(self):
        # Clean up resources or close the env (optional)
        self.env.close()


def make_env(env_args):
    def _init():
        env = CustomPyBulletEnv(env_args)
        return env
    return _init


def create_multi_envs(args, start_method='forkserver'):
    # Create the vectorized environment
    envs_func = []
    for env_id in range(args.num_envs):
        args_inst = copy.deepcopy(args)
        args_inst.seed += env_id # copy args to change its seed
        env_fc = make_env(args_inst)
        envs_func.append(env_fc)
    envs = SubprocVecEnv(envs_func, start_method=start_method)
    envs.tempENV = envs.get_attr('env')[0]
    return envs


def combine_envs_info(infos, key, env_ids=None):
    if env_ids is None: env_ids = range(len(infos))
    return [infos[id][key] for id in env_ids]


if __name__ == "__main__":
    import torch
    from argparse import ArgumentParser

    args = ArgumentParser()
    args.rendering = False
    args.debug = False
    args.asset_root = "assets"
    args.object_pool_folder = "objects/ycb_objects_origin_at_center_vhacd"
    args.num_pool_objs = 13
    args.num_placing_objs = 1
    args.random_select_pool = False
    args.random_select_placing = True
    args.default_scaling = 0.5
    args.realtime = False
    args.force_threshold = 20.
    args.vel_threshold = [1/240, np.pi/2400] # 1m/s^2 and 18 degree/s^2
    args.seed = 123456

    args.num_envs = 4

    # Create the vectorized environment
    all_envs = create_multi_envs(args, 'forkserver')
    observation = all_envs.reset()
    
    for _ in range(1000):
        random_action = torch.rand((args.num_envs, 6), device='cuda')
        observation, reward, done, info = all_envs.step(random_action)
        print(observation.shape)