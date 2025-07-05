import os
import random
import argparse
import numpy as np
import time
import gymnasium as gym
import sai_mujoco
import json
def main():
    dir_name = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    data_path = f'{dir_name}/dataset/FrankaGolfCourseEnv-v0/filtered_data_20250604_140137_joint.npz'
    ep = np.load(data_path, allow_pickle=True)
    
    env = gym.make("FrankaGolfCourseEnv-v0", render_mode="human")
    mocap = env.unwrapped.robot_model.target_mocap_id
    goal_idxs = [198, 206, 221, 227, 231, 264]
    last_i = 355

    def deterministic_int_generator(seed, low, high, count=1):
        rng = random.Random(seed)  # Create a local Random instance
        return [rng.randint(low, high) for _ in range(count)]

    goal_pos = {
        198 : [ 0.745, -0.077, 0.34 ],
        206 : [0.745, 0.003, 0.34],
        221 : [0.745, 0.003, 0.21],
        227 : [0.705, 0.003, 0.21],
        231 : [0.705, 0.003, 0.23],
        264 : [0.385, 0.003, 0.23]
    }

    sample_runs = 200
    numbers = deterministic_int_generator(seed=888, low=0, high=60000000, count=sample_runs)
    
    all_observations = []
    all_next_observations = []
    all_actions = []
    all_dones = []
    all_goals = []
    j = 0
    observation, info = env.reset(seed=numbers[j])
    
    threshold = 0.01

    for run in range(sample_runs):  
        terminated = truncated = False      
        for idx in goal_idxs:
            dist = np.inf
            while dist > threshold:
                current_pose = env.unwrapped.robot_model.data.mocap_pos[mocap]
                
                action = goal_pos[idx] - current_pose
                action = np.round(action, 3)
                action = list(np.clip(action, -0.01, 0.01))
                action += [0.0,0.0,0.0]
                action.append(ep["actions"][idx][-1])
                all_actions.append(action)
                all_goals.append(ep["observations"][idx])
                all_observations.append(observation)
                
                observation, reward, terminated, truncated, info = env.step(action)
                all_dones.append(terminated)
                all_next_observations.append(observation)
                
                dist = np.linalg.norm(goal_pos[idx] - current_pose)

        while not (terminated or truncated):
            action = [0]*6 + [-1]
            all_actions.append(action)
            all_goals.append(ep["observations"][idx])
            all_observations.append(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            all_dones.append(terminated)
            all_next_observations.append(observation)
        j+=1
        j = min(j, sample_runs-1)
        observation, info = env.reset(seed=numbers[j])
    
    output_path = f'{dir_name}/dataset/FrankaIkGolfCourseEnv-v0/train/FrankaIkGolfCourseEnv-v0_train_augmented.npz'
    np.savez_compressed(
        output_path,
        observations=np.array(all_observations, dtype=np.float32),
        actions=np.array(all_actions, dtype=np.float32),
        next_observations=np.array(all_next_observations, dtype=np.float32),
        goals=np.array(all_goals, dtype=np.float32),
        dones=np.array(all_dones, dtype=bool),
    )

    env.close()
    
if __name__ == "__main__":

    main()