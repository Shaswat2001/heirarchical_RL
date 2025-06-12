import os
import random
import argparse
import numpy as np

from agents import agents
import gymnasium as gym
import jax
import jax.numpy as jnp
from utils.buffers import buffers, Dataset
from utils.env_utils import make_env_and_datasets, make_sai_datasets 
from utils.logging import get_exp_name, setup_wandb, get_wandb_video
from utils.evaluation import *
from utils.flax_utils import save_agent, restore_agent

def main(args):

    dir_name = os.path.dirname(os.path.realpath(__file__))
    folder = "20250604_140137"
    data = np.load(f'/Users/shaswatgarg/Downloads/data/train/FrankaIkGolfCourseEnv-v0/{folder}/episodes.npy', allow_pickle=True)

    env = gym.make(args.env_name, keyframe="init_frame",render_mode="human")
    observation, info = env.reset(seed=42)

    all_obs = []
    all_actions = []
    all_terminals = []

    for ep in data:
        temp_obs = []
        temp_actions = []
        temp_terminals = []

        for i in range(len(ep["actions"])):
            action = ep["actions"][i]
            action = np.array(action)
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()

            temp_obs.append(ep["observations"][i])
            temp_actions.append(ep["actions"][i])

            done = terminated or ep["info"][i]["success"]
            temp_terminals.append(done)
            print("epsiode data: ", ep["info"][i]["success"])
            print("env data: ", terminated)
            if done:
                user_input = input("Keep this trajectory? (y/n): ").strip().lower()
                if user_input == 'y':
                    all_obs.extend(temp_obs)
                    all_actions.extend(temp_actions)
                    all_terminals.extend(temp_terminals)
                # reset trajectory buffers
                temp_obs, temp_actions, temp_terminals = [], [], []
                observation, info = env.reset()
                break

    output_path = f'{dir_name}/dataset/FrankaIkGolfCourseEnv-v0/filtered_data_{folder}.npz'
    np.savez_compressed(
        output_path,
        observations=np.array(all_obs, dtype=np.float32),
        actions=np.array(all_actions, dtype=np.float32),
        terminals=np.array(all_terminals, dtype=bool)
    )

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