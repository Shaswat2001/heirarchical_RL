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
    data_path = f'{dir_name}/dataset/FrankaGolfCourseEnv-v0/train/FrankaGolfCourseEnv-v0_train_augmented_new.npz'
    ep = np.load(data_path, allow_pickle=True)
    ep = {key: ep[key] for key in ep}
    env = gym.make("FrankaGolfCourseEnv-v0", keyframe="random", render_mode="human")
    def deterministic_int_generator(seed, low, high, count=1):
        rng = random.Random(seed)  # Create a local Random instance
        return [rng.randint(low, high) for _ in range(count)]

    seeds = [28985083, 39189165, 29870374, 33325492, 25901236, 42270151, 25147293, 36334396, 11538989, 47238118, 26383337, 59830579, 47981580, 25321605, 42846884, 29011687, 18349814, 25256078, 36226747, 720534, 57825893, 35500908, 55428395, 24976170, 26673084, 16215911, 58235089, 21037645, 17657773, 15186251, 7472467, 40834818, 58502803, 20402671, 59007249, 19133894, 43259753, 53147608, 11068875, 52072653, 54662012, 47128098, 30768360, 15850428, 13389257, 40735564, 15347279, 2800839, 50739821, 37435473, 14022423, 34746020, 32552496, 911277, 5184642, 49321153, 29919613, 55581979, 29309528, 49953776, 21132635, 18695666, 49271997, 53429113, 49783488, 34563046, 10096147, 43669297, 55680577, 2509837, 14841971, 26446839, 25810818, 33173524, 14106895, 35645491, 53041183, 20550682, 49625511, 18851595, 8547780, 4198080, 22538847, 7440840, 52103364, 35992289, 51756317]
    j = 0
    observation, info = env.reset(seed = seeds[j])
    print(list(ep["goal_idxs"]))
    numbers = deterministic_int_generator(seed=888, low=0, high=60000000, count=5)
    print(ep["actions"].shape[0])

    # for i in range(len(ep["actions"][:348])):
    #     print(f'{i} : {ep["actions"][i]}')
    for i in range(ep["actions"].shape[0]):
        action = np.array(ep["actions"][i])

        action[-1] *= 255

        if i in list(ep["goal_idxs"]):
            print(i)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated
        if terminated:
            print(j)
            j  += 1
            observation, info = env.reset(seed = seeds[j])
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('--env_module', type=str, default='sai', help='Environment (dataset) name.')
    parser.add_argument('--env_name', type=str, default='FrankaGolfCourseEnv-v0', help='Environment (dataset) name.')

    parser.add_argument('--file_name', type=str, default='filtered_data_20250604_140137_joint', help='Dataset file to load')

    args = parser.parse_args()

    main(args)