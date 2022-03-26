"""
MAIN.PY - this is the file that defines & runs the RL algorithms for Single Agent
"""

import gym
import numpy as np
import pandas as pd
import argparse
import sys
import os
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

sys.path.append("../")
from sumo_rl import SumoEnvironment
import traci

from torch import nn as nn
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

def make_env(rank):
    """
    Utility function for multiprocessed env.
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = SumoEnvironment(net_file=os.path.dirname(__file__)+'/../nets/2way-single-intersection/single-intersection.net.xml',
                            route_file=os.path.dirname(__file__)+'/../nets/2way-single-intersection/single-intersection-vhvh.rou.xml',
                            out_csv_name='output/waiting_times{}'.format(rank),
                            single_agent=True,
                            use_gui=False,
                            num_seconds=100000,
                            max_depart_delay=0)
        return env
    return _init

if __name__ == '__main__':
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Args for Env""")
    prs.add_argument("-method", dest="method", type=str, default='dqn', required=False, help="which file to run.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.001, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-gamma", dest="gamma", type=float, default=0.99, required=False, help="discount factor.\n")
    prs.add_argument("-st", dest="steps", type=int, default=2048, required=False, help="n steps.\n")
    prs.add_argument("-batch", dest="batch_size", type=int, default=128, required=False, help="batch_size. \n")
    prs.add_argument("-cr", dest="clip_range", type=float, default=0.2, required=False, help="clip_range. \n")
    prs.add_argument('-vfc', dest="vf_coef", type=float, default=0.5, required=False, help="vf coef. \n")
    prs.add_argument('-efc', dest="ent_coef", type=float, default=0.0, required=False, help="ent coef. \n")
    prs.add_argument('-maxgrad', dest="max_grad_norm", type=float, default=0.5, required=False, help="max grad norm \n")
    args = prs.parse_args()
   
    num_cpu = 8  # Number of processes to use
    
    if args.method == 'ppo':
        # Create the vectorized environment
        env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

        model = PPO('MlpPolicy', env, gamma=0.99, learning_rate=0.0005, n_steps=128, n_epochs=20,
                    batch_size=256, clip_range=0.2,verbose=0)

        model.learn(total_timesteps=800000) 
                
    elif args.method == 'a2c':
        # Create the vectorized environment
        env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        model = A2C("MlpPolicy", env, gamma=0.99, learning_rate=0.0005, n_steps=5, verbose=0)
        
        model.learn(total_timesteps=800000)
    elif args.method == 'dqn':
        # Create the vectorized environment
        env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        model = DQN("MlpPolicy", env, gamma=0.99, learning_rate=0.0005, verbose=0)
        
        model.learn(total_timesteps=800000)
    else:
        env = SumoEnvironment(net_file=os.path.dirname(__file__)+'/../nets/2way-single-intersection/single-intersection.net.xml',
                            route_file=os.path.dirname(__file__)+'/../nets/2way-single-intersection/single-intersection-vhvh.rou.xml',
                            out_csv_name='output/waiting_times',
                            single_agent=True,
                            use_gui=False,
                            num_seconds=100000,
                            max_depart_delay=0)
        if args.method == "random":
            env.reset()
            for i in range(100000):
                action = env.action_space.sample()
                obs, reward, done, _ = env.step(action)
                if done:
                    env.reset()
        elif args.method == "fixed":
            obs = env.reset()
            action = 0
            for i in range(60000):
                for j in range(2):
                    obs, reward, done, _ = env.step(action)
                    if done:
                        env.reset()
                action +=1
                action %= 4
            env.reset()
        else:
            print('Invalid algorithm')
