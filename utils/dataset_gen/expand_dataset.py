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
    data_path = f'{dir_name}/dataset/FrankaGolfCourseEnv-v0/train/FrankaGolfCourseEnv-v0_train_augmented.npz'
    ep = np.load(data_path, allow_pickle=True)
    ep = {key: ep[key] for key in ep}
    env = gym.make("FrankaGolfCourseEnv-v0", keyframe="random", render_mode="human")
    def deterministic_int_generator(seed, low, high, count=1):
        rng = random.Random(seed)  # Create a local Random instance
        return [rng.randint(low, high) for _ in range(count)]

    def interpolate_by_distance(start, end, threshold):
        """Discretizes a straight line from start to end based on distance threshold."""
        diff = end - start
        dist = np.linalg.norm(diff)
        num_points = max(2, int(np.ceil(dist / threshold)) + 1)
        return np.linspace(start, end, num_points)

    rest_observations = ep["observations"][:188]
    rest_actions = ep["actions"][:188]
    rest_dones = ep["terminals"][:188]

    keys = [7, 16, 27, 36, 42, 56, 187]

    ep["goal_idxs"] = np.array([7, 16, 27, 36, 42, 56, 187], np.int32)
    seeds = deterministic_int_generator(888, 1, 60000000, 100)
    print(seeds)
    for i in range(100):
        observation, info = env.reset(seed=seeds[i])

        threshold = 0.05
        interp_obs = interpolate_by_distance(observation, ep["observations"][0], threshold)  # Shape (N, obs_dim)

        new_obs = []
        new_actions = []
        new_dones = []
        for i in range(interp_obs.shape[0]):
            new_obs.append(observation)
            action = list(interp_obs[i][:7]) + [1.0]
            new_actions.append(action)
            new_dones.append(False)

            action[-1] = 255
            observation, reward, terminated, truncated, info = env.step(action)

        keep = input("Do you want to keep this trajectory")

        if keep == "Y":
            new_obs = np.array(new_obs, np.float32)
            new_actions = np.array(new_actions, np.float32)
            new_dones = np.array(new_dones, bool)

            new_obs = np.concatenate([new_obs, rest_observations], axis=0)
            new_actions = np.concatenate([new_actions, rest_actions], axis=0)
            new_dones = np.concatenate([new_dones, rest_dones], axis=0)

            new_keys = [ep["terminals"].shape[-1] - 1 + interp_obs.shape[0]]  # the new episode starts at 0
            append_keys = [ep["terminals"].shape[-1] + interp_obs.shape[0] + k for k in keys]

            ep["observations"] = np.concatenate([ep["observations"], new_obs], axis = 0)
            ep["actions"] = np.concatenate([ep["actions"], new_actions], axis = 0)
            ep["terminals"] = np.concatenate([ep["terminals"], new_dones], axis = 0)
            
            ep["goal_idxs"] = np.concatenate([ep["goal_idxs"], np.array(new_keys+append_keys)], axis = 0)

            output_path = f'{dir_name}/dataset/FrankaGolfCourseEnv-v0/train/FrankaGolfCourseEnv-v0_train_augmented_new.npz'
            np.savez_compressed(
                output_path,
                observations=ep["observations"],
                actions=ep["actions"],
                # next_observations=np.array(all_next_observations, dtype=np.float32),
                goal_idxs=ep["goal_idxs"],
                terminals=ep["terminals"],
            )

            print(new_obs.shape)    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('--env_module', type=str, default='sai', help='Environment (dataset) name.')
    parser.add_argument('--env_name', type=str, default='FrankaGolfCourseEnv-v0', help='Environment (dataset) name.')

    parser.add_argument('--file_name', type=str, default='filtered_data_20250604_140137_joint', help='Dataset file to load')

    args = parser.parse_args()

    main(args)