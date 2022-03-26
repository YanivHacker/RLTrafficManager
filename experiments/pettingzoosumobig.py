"""
Here we define & run the RL algorithms in the 3*3 intersection grid
"""

from stable_baselines3 import PPO, A2C, DQN
from pantheonrl.common.agents import OnPolicyAgent
from pantheonrl.envs.pettingzoo import PettingZooAECWrapper
import os
import sys
import argparse
from sumo_rl import env
import traci

if __name__ == '__main__':
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Args for Env""")
    prs.add_argument("-method", dest="method", type=str, default='ppo', required=False, help="which file to run.\n")
    args = prs.parse_args()

    env = PettingZooAECWrapper(env(net_file=os.path.dirname(__file__)+'/../nets/3x3grid/3x3Grid2lanes.net.xml', #double/network.net.xml
                    route_file=os.path.dirname(__file__)+'/../nets/3x3grid/routes14000.rou.xml', #double/flow.rou.xml
                    use_gui=True,
                    num_seconds=14000,
                    out_csv_name='output/waiting_times_pettingzoo')
            )

    print(env.n_players)
    for i in range(env.n_players - 1):
        partner = OnPolicyAgent(PPO('MlpPolicy', env.getDummyEnv(i), verbose=1))
        env.add_partner_agent(partner, player_num=i + 1)

    if args.method == "ppo":
        ego = PPO('MlpPolicy', env, gamma=0.99, learning_rate=0.0005, n_steps=128, n_epochs=20,
                            clip_range=0.2, verbose=0)
        ego.learn(total_timesteps=100000)
    elif args.method == "a2c":
        ego = A2C("MlpPolicy", env, gamma=0.99, learning_rate=0.0005, n_steps=5, verbose=0)
        ego.learn(total_timesteps=100000)
    elif args.method == "dqn":
        ego = DQN('MlpPolicy', env, gamma=0.99, learning_rate=0.0005, verbose=0)
        ego.learn(total_timesteps=100000)
    elif args.method == "random":
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
        print('Invalid choice')

    
