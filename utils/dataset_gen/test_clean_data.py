import os
import random
import argparse
import numpy as np

import gymnasium as gym
import sai_mujoco

def main(args):

    dir_name = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    ep = np.load(f'{dir_name}/dataset/FrankaGolfCourseEnv-v0/train/FrankaGolfCourseEnv-v0_train.npz', allow_pickle=True)
    print(ep["observations"].shape)
    output_path = f'{dir_name}/dataset/FrankaGolfCourseEnv-v0/train/FrankaGolfCourseEnv-v0_train_robot.npz'
    np.savez_compressed(
        output_path,
        observations=ep["observations"][:,:18],
        actions=ep["actions"],
        # joint_angles=np.array(all_joint_angles, dtype=np.float32),
        terminals=ep["terminals"]
    )
    # env = gym.make(args.env_name, keyframe="init_frame",render_mode="human")
    
    # observation, info = env.reset(seed=42)
    # for i in range(len(ep["actions"])):
    #     action = ep["actions"][i]
    #     action = np.array(action)
    #     action[-1] *= 255
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     env.render()

    #     done = ep["terminals"][i]
    #     if done:
    #         # user_input = input("Keep this trajectory? (y/n): ").strip().lower()
    #         observation, info = env.reset()

    # env.close()
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('--env_module', type=str, default='sai', help='Environment (dataset) name.')
    parser.add_argument('--env_name', type=str, default='FrankaGolfCourseEnv-v0', help='Environment (dataset) name.')

    parser.add_argument('--file_name', type=str, default='filtered_data_20250604_140137_joint', help='Dataset file to load')

    args = parser.parse_args()

    main(args)