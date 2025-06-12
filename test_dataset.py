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

    data = np.load('/home/ubuntu/uploads/heirarchical_RL/dataset/FrankaIkGolfCourseEnv-v0/train/FrankaIkGolfCourseEnv-v0_train.npz', allow_pickle=True)

    # List all arrays stored in the file
    print("Keys in the .npz file:", data.files)

    # Access each array by key
    for key in data.files:
        print(f"{key}: shape = {data[key].shape}, dtype = {data[key].dtype}")
        print(data[key])  # Print actual data if needed

    for i in data["actions"]:
        print(i)
    env = gym.make(args.env_name, keyframe="init_frame",render_mode="human")
    observation, info = env.reset(seed=42)
    rwd = []
    for i in range(len(data["observations"])):
        action = data["actions"][i]
        print(action)
        action = np.array(action)
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
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