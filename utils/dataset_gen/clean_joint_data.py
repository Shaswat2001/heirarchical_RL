import os
import random
import argparse
import numpy as np

import gymnasium as gym
import sai_mujoco

def main(args):

    dir_name = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    folder = "20250604_140137"
    data = np.load(f'{dir_name}/dataset/FrankaGolfCourseEnv-v0/filtered_data_{folder}.npz', allow_pickle=True)

    env = gym.make(args.env_name, keyframe="init_frame",render_mode="human")
    observation, info = env.reset(seed=42)

    all_obs = []
    all_actions = []
    all_joint_angles = []
    all_terminals = []
    temp_obs = []
    temp_actions = []
    temp_joint_angles = []
    temp_terminals = []
    for i in range(data["joint_angles"].shape[0]):
        action = data["joint_angles"][i]
        action = np.array(action)
        action[-1] *= 255
        observation, reward, terminated, truncated, info = env.step(action)
        
        temp_obs.append(data["observations"][i])
        temp_actions.append(data["actions"][i])
        temp_joint_angles.append(data["joint_angles"][i])

        temp_terminals.append(data["terminals"][i])
        if data["terminals"][i]:
            print(i)
            user_input = input("Keep this trajectory? (y/n): ").strip().lower()
            if user_input == 'y':
                all_obs.extend(temp_obs)
                all_actions.extend(temp_actions)
                all_terminals.extend(temp_terminals)
                all_joint_angles.extend(temp_joint_angles)
            # reset trajectory buffers
            temp_obs, temp_actions, temp_terminals, temp_joint_angles = [], [], [], []
            observation, info = env.reset()

    output_path = f'{dir_name}/dataset/FrankaGolfCourseEnv-v0/filtered_data_{folder}_joint.npz'
    np.savez_compressed(
        output_path,
        observations=np.array(all_obs, dtype=np.float32),
        actions=np.array(all_actions, dtype=np.float32),
        joint_angles=np.array(all_joint_angles, dtype=np.float32),
        terminals=np.array(all_terminals, dtype=bool)
    )

    env.close()
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('--env_module', type=str, default='sai', help='Environment (dataset) name.')
    parser.add_argument('--env_name', type=str, default='FrankaGolfCourseEnv-v0', help='Environment (dataset) name.')

    args = parser.parse_args()

    main(args)