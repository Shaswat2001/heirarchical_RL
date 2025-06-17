import random
import argparse
import numpy as np

from agents import agents
import gymnasium as gym
import ogbench
import cv2
import jax
from utils.buffers import buffers, Dataset
from utils.env_utils import make_env_and_datasets, make_sai_datasets 
from utils.logging import get_exp_name
from utils.evaluation import *
from utils.flax_utils import restore_agent

def frames_to_video(frames, output_path="output.mp4", fps=30):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert if frames are RGB
        out.write(frame_bgr)

    out.release()

def main(args):

    exp_name = get_exp_name(args.env_name, args.env_name)

    if args.env_module == "ogbench":
        env, train_dataset, _ = make_env_and_datasets(args.env_name)
    else:
        env, train_dataset, _ = make_sai_datasets(args.env_name)
        
    random.seed(args.seed)
    np.random.seed(args.seed)

    (agent_class, agent_config) = agents[args.agents]
    render = []
    buffer_class = buffers[agent_config["dataset_class"]]
    train_dataset = buffer_class(Dataset.create(**train_dataset),agent_config)
    example_batch = train_dataset.sample(1)

    agent = agent_class.create(
        args.seed,
        example_batch['observations'],
        example_batch['actions'],
        {},
    )

    splits = args.env_name.split('-')
    env_name = '-'.join(splits[:-2] + splits[-1:])
    agent = restore_agent(agent, args.restore_path, args.restore_epoch)
    agent = supply_rng(agent.get_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    env = gym.make(env_name)
    observation, info = env.reset(options=dict(task_id=1, render_goal=True))
    rwd = []
    goal = info["goal"]
    i = 0
    while True:
        action = agent(observation=observation, goal=goal, temperature=0.0)
        action = np.array(action)
        observation, reward, terminated, truncated, info = env.step(action)
        frame = env.render().copy()
        render.append(frame)
        rwd.append(reward)
        # env.render()
        if terminated or truncated:
            frames_to_video(render)
            i = 0
            observation, info = env.reset()
        i += 1

    env.close()

    print(rwd)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_group', type=str, default='Debug', help='Run group.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--agents', type=str, default="hiql", help='Agent to load.')

    # Environment
    parser.add_argument('--env_module', type=str, default='ogbench', help='Environment (dataset) name.')
    parser.add_argument('--env_name', type=str, default='antmaze-medium-navigate-v0', help='Environment (dataset) name.')

    # Save / restore
    parser.add_argument('--restore_path', type=str, default='exp/hrl-arenaX/Debug/antmaze_20250614-230705_hiql', help='Save directory.')
    parser.add_argument('--restore_epoch', type=int, default=900000, help='Epoch checkpoint.')

    args = parser.parse_args()

    main(args)