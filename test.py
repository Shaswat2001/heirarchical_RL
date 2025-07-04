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
    agent = supply_rng(agent.get_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    env = gym.make(args.env_name, keyframe="random",render_mode="human")
    observation, info = env.reset(seed=42)
    rwd = []
    with np.load(f'/home/ubuntu/uploads/heirarchical_RL/dataset/FrankaIkGolfCourseEnv-v0/filtered_data_20250604_140137.npz', allow_pickle=True) as data:
        train_data = {key: data[key] for key in data}
    print(np.linalg.norm(observation - info.get("goal")))
    j = [35, 40, 58, 65, 70, 75, 81,  94, 158]
    threshold = [0.35, 0.2, 0.4, 0.3, 0.25, 0.25, 0.25, 0.25, 0.2]
    i = 0
    k =0
    import keras
    print(env.unwrapped.keyframe)
    model = keras.models.load_model("converted_tf1_model.keras")
    while True:
        input = np.hstack([observation, train_data["observations"][i+1]]).reshape(1,-1)
        action = model(input)
        action = np.array(action)[0]

        dist = np.linalg.norm(observation - train_data["observations"][i+1])
        print(dist)
        print(i)
        action[:-1] = action[:-1]*0.01

        # action[-1] *= 255
        observation, reward, terminated, truncated, info = env.step(action)
        rwd.append(reward)
        env.render()

        if dist < 0.4:
            i += 1
            i = min(i, train_data["observations"].shape[0] - 1)
        # if i < 80:
        #     i += 1

        if terminated or truncated:
            i = 0
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
    parser.add_argument('--restore_path', type=str, default='exp/hrl-arenaX/Debug/FrankaIkGolfCourseEnv_20250612-235606_gcbc', help='Save directory.')
    parser.add_argument('--restore_epoch', type=int, default=400000, help='Epoch checkpoint.')

    args = parser.parse_args()

    main(args)