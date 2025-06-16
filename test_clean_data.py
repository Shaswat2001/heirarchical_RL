import os
import random
import argparse
import numpy as np

from agents import agents
import gymnasium as gym

def main(args):

    dir_name = os.path.dirname(os.path.realpath(__file__))
    ep = np.load(f'/Users/shaswatgarg/Documents/Job/ArenaX/Development/heirarchical_RL/dataset/FrankaIkGolfCourseEnv-v0/filtered_data_20250604_135129.npz', allow_pickle=True)

    env = gym.make(args.env_name, keyframe="init_frame",render_mode="human")
    observation, info = env.reset(seed=42)

    for i in range(len(ep["actions"])):
        action = ep["actions"][i]
        action = np.array(action)
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()

        done = ep["terminals"][i]
        print("epsiode state data: ", ep["observations"][i])
        print("epsiode done data: ", ep["terminals"][i])
        print("epsiode action data: ", ep["actions"][i])
        if done:
            user_input = input("Keep this trajectory? (y/n): ").strip().lower()
            observation, info = env.reset()

    env.close()
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_group', type=str, default='Debug', help='Run group.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--agents', type=str, default="gcbc", help='Agent to load.')

    # Environment
    parser.add_argument('--env_module', type=str, default='sai', help='Environment (dataset) name.')
    parser.add_argument('--env_name', type=str, default='FrankaIkGolfCourseEnv-v0', help='Environment (dataset) name.')

    # Save / restore
    parser.add_argument('--save_dir', type=str, default='exp/', help='Save directory.')
    parser.add_argument('--save_epoch', type=int, default=100000, help='Epoch checkpoint.')

    args = parser.parse_args()

    main(args)