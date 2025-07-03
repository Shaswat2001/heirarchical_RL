import os
import random
import argparse
import numpy as np
import time
import gymnasium as gym
import sai_mujoco
import json
def main(args):
    dir_name = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    data_path = '/Users/shaswatgarg/Documents/Job/ArenaX/Development/heirarchical_RL/dataset/FrankaIkGolfCourseEnv-v0/filtered_data_20250604_140137.npz'
    ep = np.load(data_path, allow_pickle=True)
    env = gym.make("FrankaIkGolfCourseEnv-v0", keyframe="init_frame", render_mode="human")

    all_observations = []
    all_actions = []
    all_next_observations = []
    all_rewards = []
    all_dones = []
    # j = [25, 58, 81,  94, 158]
    # selected_obs = {str(idx): ep["observations"][idx].tolist() for idx in j}
    # json_path = os.path.join(dir_name, "selected_observations.json")
    # with open(json_path, 'w') as f:
    #     json.dump(selected_obs, f, indent=4)
    print(ep["terminals"][:348])
    observation, info = env.reset(seed=42)
    j = 0
    for i in range(len(ep["actions"])):
        action = np.array(ep["actions"][i])

        all_observations.append(observation)
        all_actions.append(action)
        y = input("HI")
        print(j)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated
        all_next_observations.append(next_observation)
        all_rewards.append(reward)
        all_dones.append(done)

        observation = next_observation

        if terminated or truncated:
            j =0
            observation, info = env.reset()
        j += 1
    observation, info = env.reset()

    env.close()
    # print(all_observations)
    # output_path = f'{dir_name}/dataset/FrankaGolfCourseEnv-v0/train/FrankaGolfCourseEnv-v0_train_augmented.npz'
    # np.savez_compressed(
    #     output_path,
    #     observations=np.array(all_observations, dtype=np.float32),
    #     actions=np.array(all_actions, dtype=np.float32),
    #     next_observations=np.array(all_next_observations, dtype=np.float32),
    #     rewards=np.array(all_rewards, dtype=np.float32),
    #     dones=np.array(all_dones, dtype=bool),
    # )
    # print(f"Saved augmented dataset to {output_path}")
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('--env_module', type=str, default='sai', help='Environment (dataset) name.')
    parser.add_argument('--env_name', type=str, default='FrankaGolfCourseEnv-v0', help='Environment (dataset) name.')

    parser.add_argument('--file_name', type=str, default='filtered_data_20250604_140137_joint', help='Dataset file to load')

    args = parser.parse_args()

    main(args)