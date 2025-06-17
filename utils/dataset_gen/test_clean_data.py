import os
import random
import argparse
import numpy as np

import gymnasium as gym
import sai_mujoco

def main(args):

    dir_name = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    ep = np.load(f'{dir_name}/dataset/FrankaGolfCourseEnv-v0/{args.file_name}_final.npz', allow_pickle=True)
    # ep = dict(ep)
    # ep.pop('actions', None)  # avoids KeyError if the key doesn't exist

    # # Rename a key
    # ep['actions'] = ep.pop('joint_angles')
    
    # np.savez_compressed(
    #     f'{dir_name}/dataset/FrankaGolfCourseEnv-v0/{args.file_name}_final.npz',
    #     **ep
    # )
    env = gym.make(args.env_name, keyframe="init_frame",render_mode="human")
    observation, info = env.reset(seed=42)
    for i in range(len(ep["actions"])):
        action = ep["actions"][i]
        action = np.array(action)
        action[-1] *= 255
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()

        done = ep["terminals"][i]
        if done:
            user_input = input("Keep this trajectory? (y/n): ").strip().lower()
            observation, info = env.reset()

    env.close()
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('--env_module', type=str, default='sai', help='Environment (dataset) name.')
    parser.add_argument('--env_name', type=str, default='FrankaGolfCourseEnv-v0', help='Environment (dataset) name.')

    parser.add_argument('--file_name', type=str, default='filtered_data_20250604_140137_joint', help='Dataset file to load')

    args = parser.parse_args()

    main(args)