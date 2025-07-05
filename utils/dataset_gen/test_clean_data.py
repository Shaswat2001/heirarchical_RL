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
    data_path = '/Users/shaswatgarg/Documents/Job/ArenaX/Development/heirarchical_RL/dataset/FrankaGolfCourseEnv-v0/train/FrankaIkGolfCourseEnv-v0_train_augmented.npz'
    ep = np.load(data_path, allow_pickle=True)
    env = gym.make("FrankaGolfCourseEnv-v0", keyframe="init_frame", render_mode="human")
    def deterministic_int_generator(seed, low, high, count=1):
        rng = random.Random(seed)  # Create a local Random instance
        return [rng.randint(low, high) for _ in range(count)]

    # mocap = env.unwrapped.robot_model.target_mocap_id
    all_observations = []
    all_actions = []
    # all_next_observations = []
    # all_rewards = []
    all_dones = []
    # j = [25, 58, 81,  94, 158]
    # selected_obs = {str(idx): ep["observations"][idx].tolist() for idx in j}
    # json_path = os.path.join(dir_name, "selected_observations.json")
    # with open(json_path, 'w') as f:
    #     json.dump(selected_obs, f, indent=4)
    # print(ep["terminals"][:348])
    observation, info = env.reset(seed=42)
    numbers = deterministic_int_generator(seed=888, low=0, high=60000000, count=5)
    j = 0
    print(len(ep["actions"]))
    # for i in range(len(ep["actions"][:348])):
    #     print(f'{i} : {ep["actions"][i]}')
    for i in range(348):
        action = np.array(ep["actions"][i])

        action[-1] *= 255
        # print(observation)
        # print(ep["observations"][i])
        # input()
        all_observations.append(observation)
        all_actions.append(ep["actions"][i])
        # if i in [198, 206, 221, 227, 231, 264]:
        #     print(env.unwrapped.robot_model.data.mocap_pos[mocap])
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated
        # all_next_observations.append(next_observation)
        # all_rewards.append(reward)
        all_dones.append(done)

        if terminated or truncated:
            print(j)
            j =0
            observation, info = env.reset(seed=42)
        j += 1
    # observation, info = env.reset()
    print(True in all_dones)
    env.close()
    # print(all_observations)
    output_path = f'{dir_name}/dataset/FrankaGolfCourseEnv-v0/train/FrankaGolfCourseEnv-v0_train_augmented.npz'
    np.savez_compressed(
        output_path,
        observations=np.array(all_observations, dtype=np.float32),
        actions=np.array(all_actions, dtype=np.float32),
        # next_observations=np.array(all_next_observations, dtype=np.float32),
        # rewards=np.array(all_rewards, dtype=np.float32),
        dones=np.array(all_dones, dtype=bool),
    )
    # print(f"Saved augmented dataset to {output_path}")
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('--env_module', type=str, default='sai', help='Environment (dataset) name.')
    parser.add_argument('--env_name', type=str, default='FrankaGolfCourseEnv-v0', help='Environment (dataset) name.')

    parser.add_argument('--file_name', type=str, default='filtered_data_20250604_140137_joint', help='Dataset file to load')

    args = parser.parse_args()

    main(args)