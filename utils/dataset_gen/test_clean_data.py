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

    env = gym.make("FrankaGolfCourseEnv-v0", keyframe="random", render_mode="human")
    # def deterministic_int_generator(seed, low, high, count=1):
    #     rng = random.Random(seed)  # Create a local Random instance
    #     return [rng.randint(low, high) for _ in range(count)]

    seeds = [15249118404380085639, 11160733087153706170, 5574901013384822076, 14975582312850916589, 
             6935404696938344435, 17969950218656060885, 2517673399058375730, 17745886526460971218, 
             14637512743872356621, 17221282304188126126, 9009127414828977831, 2735056578005115762, 
             17935682492066825726, 8432121832776208629, 2320789556007449690, 12984577411975327593, 
             10070027512249063798, 4299579401295960059, 10211518166902230745, 8198621770309551693, 
             9288563261616740603, 12012270746048031957, 13354490029136460318, 8366392630590506886, 
             2239100903254005056, 1041429938100427011, 11531536054697045965, 11099008592770530883, 
             11651407931918419824, 15058861449373575226, 362089145899586923, 1032320380402351332, 
             15010693726868240756, 10031381962380992446, 13616136326427811631, 1692165768332727562, 
             2108839775597975821, 3206485344533219578, 18303973879305603268, 12814296373191120281, 
             10765987620087006141, 2742203196291721289, 13683262468274108609, 10134042530815520126, 
             8637179986327094643, 9227324760360540502, 973398537799208969, 2967301329522480460, 
             5692930283327308413, 968899172769838687, 10327759251782185725, 12699618903397884819, 
             2704835705367505262, 15780812269047924098, 16361280639725058175, 12650951613230308876, 
             2323442919270060514, 17906437972804021813, 6624016450053325651, 5088467518978949889, 
             5666511788609898502, 7314552822875358124, 5117368381706122026, 1145860449695299371, 
             3135944470876044986, 13967646923563520065, 5843985234160177859, 18159122152459654418, 
             8737664961353780714]
    j = 0
    observation, info = env.reset(seed = seeds[j])
    print(list(ep["goal_idxs"]))
    # numbers = deterministic_int_generator(seed=888, low=0, high=60000000, count=5)
    # print(ep["actions"].shape[0])

    # for i in range(len(ep["actions"][:348])):
    #     print(f'{i} : {ep["actions"][i]}')
    for i in range(353, ep["actions"].shape[0]):
        action = np.array(ep["actions"][i])

        action[-1] *= 255
        observation, reward, terminated, truncated, info = env.step(action)

        if i in list(ep["goal_idxs"]):
            input(f"{i}")
        done = terminated
        if ep["terminals"][i]:
            # print(j)
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