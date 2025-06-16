import os
import random
import argparse
import numpy as np

import gymnasium as gym
import sai_mujoco

def main(args):

    dir_name = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    folder = "20250604_140137"
    data = np.load(f'/Users/shaswatgarg/Downloads/data/train/FrankaIkGolfCourseEnv-v0/{folder}/episodes.npy', allow_pickle=True)

    env = gym.make(args.env_name, keyframe="init_frame",render_mode="human")
    observation, info = env.reset(seed=42)

    all_obs = []
    all_actions = []
    all_joint_angles = []
    all_terminals = []

    for ep in data:
        temp_obs = []
        temp_actions = []
        temp_joint_angles = []
        temp_terminals = []

        for i in range(len(ep["actions"])):
            action = ep["actions"][i]
            action = np.array(action)

            observation, reward, terminated, truncated, info = env.step(action)
            # env.render()
            if i < 125 and (ep["actions"][i] == ep["actions"][i+1]).all() and (ep["actions"][i][:-1] == np.zeros((6))).all():
                print("TRUE")
                continue
            
            joint_angles = info["joint_angles"].copy()
            joint_angles[-1] = joint_angles[-1] / 255
            temp_obs.append(ep["observations"][i])
            temp_actions.append(ep["actions"][i])
            temp_joint_angles.append(joint_angles)

            done = terminated or ep["info"][i]["success"]
            temp_terminals.append(done)
            print("epsiode data: ", ep["info"][i]["success"])
            print("env data: ", terminated)
            if done:
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
                break

    output_path = f'{dir_name}/dataset/FrankaGolfCourseEnv-v0/filtered_data_{folder}.npz'
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
    parser.add_argument('--env_name', type=str, default='FrankaIkGolfCourseEnv-v0', help='Environment (dataset) name.')

    args = parser.parse_args()

    main(args)