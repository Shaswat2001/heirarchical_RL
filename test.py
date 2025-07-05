import random
import argparse
import numpy as np
import pickle
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
    env = gym.make(args.env_name, keyframe="init_frame",render_mode="human")
    observation, info = env.reset(seed=42)
    rwd = []
    with np.load(f'/home/ubuntu/uploads/heirarchical_RL/dataset/FrankaGolfCourseEnv-v0/train/FrankaGolfCourseEnv-v0_train_augmented.npz', allow_pickle=True) as data:
        train_data = {key: data[key] for key in data}

    j = [7, 16, 27, 36, 42, 56, 187]
    threshold = [0.1, 0.25, 0.45, 0.37, 0.1, 0.1, 0.1]
    # threshold = [4, 6, 6, 6, 6, 6, 6]
    i = 0
    k =0
    # import keras
    # print(env.unwrapped.keyframe)
    # model = keras.models.load_model("converted_tf1_model.keras")
    with open('standard_scaler.pkl', 'rb') as f:
        loaded_scaler = pickle.load(f)
    while True:
        # input = np.hstack([observation, train_data["observations"][i+1]]).reshape(1,-1)
        action = agent(observation=observation, goal=train_data["observations"][j[i]], temperature=0.0)
        # action = model(input)
        action = np.array(action)
        dist = np.linalg.norm(observation - train_data["observations"][j[i]])
        
        action = loaded_scaler.inverse_transform(action.reshape(1, -1))[0]
        action[-1] *= 255
        print(i)
        print(dist)
        # input()
        k += 1
        # action = np.clip(action, -1, 1)
        # action[:-1] = action[:-1]*0.01

        # sub_action = action[:-1]

        # # Find index of max absolute value
        # max_idx = np.argmax(np.abs(sub_action))

        # # Create new sub-action with all zeros
        # new_sub_action = np.zeros_like(sub_action)

        # # Set only the max index to ±0.01 based on sign
        # new_sub_action[max_idx] = 0.01 if sub_action[max_idx] > 0 else -0.01

        # # Assign back to the original action array
        # action[:-1] = new_sub_action

        # action[-1] *= 255

        observation, reward, terminated, truncated, info = env.step(action)
        rwd.append(reward)
        env.render()

        if dist < threshold[i]:
            i += 1
            i = min(i, len(threshold)-1)
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
    parser.add_argument('--env_name', type=str, default='FrankaGolfCourseEnv-v0', help='Environment (dataset) name.')

    # Save / restore
    parser.add_argument('--restore_path', type=str, default='exp/hrl-arenaX/Debug/FrankaGolfCourseEnv_20250705-173614_gcbc', help='Save directory.')
    parser.add_argument('--restore_epoch', type=int, default=0, help='Epoch checkpoint.')

    args = parser.parse_args()

    main(args)