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

    exp_name = get_exp_name(args.env_name, args.env_name)

    if args.env_module == "ogbench":
        env, train_dataset, _ = make_env_and_datasets(args.env_name)
    else:
        env, train_dataset, _ = make_sai_datasets(args.env_name)
        
    random.seed(args.seed)
    np.random.seed(args.seed)

    (agent_class, agent_config) = agents[args.agents]

    buffer_class = buffers[agent_config["dataset_class"]]
    train_dataset = buffer_class(Dataset.create(**train_dataset),agent_config)
    example_batch = train_dataset.sample(1)

    agent = agent_class.create(
        args.seed,
        example_batch['observations'],
        example_batch['actions'],
        {},
    )

    agent = restore_agent(agent, args.restore_path, args.restore_epoch)
    env = gym.make("HumanoidWalkEnv-v0", render_mode="human")
    observation, info = env.reset(seed=42)
    rwd = []
    for _timestep in range(100000):
        action = agent(observation=observation, goal=info.get("goal"), temperature=0.0)
        action = np.array(action)
        observation, reward, terminated, truncated, info = env.step(action)
        rwd.append(reward)
        env.render()

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

    print(rwd)
    
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