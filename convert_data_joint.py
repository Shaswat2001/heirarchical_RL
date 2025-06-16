import os
import random
import argparse
import numpy as np

import copy
from agents import agents
import gymnasium as gym
import sai_mujoco

def main(args):

    input_dir = "/Users/shaswatgarg/Documents/Job/ArenaX/Development/heirarchical_RL/dataset/FrankaIkGolfCourseEnv-v0/train"  # <-- change this
    env = gym.make(args.env_name, keyframe="init_frame")
    # List all .npz files in the folder
    npz_files = [f for f in os.listdir(input_dir) if f.endswith(".npz")]
    print(npz_files)
    npz_files = ["FrankaIkGolfCourseEnv-v0_train_joint.npz"]
    for filename in npz_files:
        filepath = os.path.join(input_dir, filename)
        ep = np.load(filepath, allow_pickle=True)
        joint_dataset = {}
        joint_dataset["actions"] = []
        joint_dataset["terminals"] = copy.deepcopy(ep["terminals"])
        joint_dataset["observations"] = copy.deepcopy(ep["observations"])

        observation, info = env.reset(seed=42)
        for i in range(len(ep["actions"])):
            
            action = ep["actions"][i]
            action = np.array(action)
            observation, reward, terminated, truncated, info = env.step(action)
            joint_val = env.unwrapped.robot_model.data.ctrl.copy()
            joint_val[-1] = 1.0 if joint_val[-1] > 0 else 0.0
            joint_dataset["actions"].append(joint_val)
            done = ep["terminals"][i]
            if done:
                observation, info = env.reset()

        env.close()
        # print(joint_dataset["actions"])
        joint_dataset["actions"] = np.array(joint_dataset["actions"],dtype=np.float32)
        # print(joint_dataset["actions"].shape)
        a = 2.*(joint_dataset["actions"] - np.min(joint_dataset["actions"]))/np.ptp(joint_dataset["actions"])-1
        print(a)
        # print(joint_dataset["actions"])
        print(np.max(a))
        print(np.min(a))
        # print(joint_dataset["terminals"].shape)
        # print(joint_dataset["observations"].shape)
        output_path = os.path.join(input_dir, f"{filename}_joint.npz")
        # np.savez_compressed(output_path, **joint_dataset)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_group', type=str, default='Debug', help='Run group.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--agents', type=str, default="gcbc", help='Agent to load.')

    # Environment
    parser.add_argument('--env_module', type=str, default='sai', help='Environment (dataset) name.')
    parser.add_argument('--env_name', type=str, default='FrankaIkGolfCourseEnv-v0', help='Environment (dataset) name.')

    # Save / restore
    parser.add_argument('--save_dir', type=str, default='exp/', help='Save directory.')
    parser.add_argument('--save_epoch', type=int, default=100000, help='Epoch checkpoint.')

    args = parser.parse_args()

    main(args)