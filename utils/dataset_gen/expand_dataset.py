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
    seeds = [15249118404380085639, 9135532071002051198, 11160733087153706170, 5574901013384822076, 8626708206972141402, 17398021499301043337, 14975582312850916589, 8091316526931800035, 6935404696938344435, 
            17969950218656060885, 2517673399058375730, 17745886526460971218, 14637512743872356621, 17221282304188126126, 8598563967139042966, 12097468403053798984, 9009127414828977831, 4309739594016933652, 
            2735056578005115762, 5684417553841107255, 17935682492066825726, 8432121832776208629, 2320789556007449690, 12984577411975327593, 10070027512249063798, 4299579401295960059, 10211518166902230745, 
            9745686507364506787, 8198621770309551693, 9288563261616740603, 12012270746048031957, 13354490029136460318, 8366392630590506886, 15912983976522937564, 7087934259486051664, 17783477137488645394, 
            3552441185365854992, 2239100903254005056, 14328556416238174830, 1041429938100427011, 11531536054697045965, 12344179143464617669, 11099008592770530883, 11651407931918419824, 15797627341532818634, 
            1684009603494821013, 15058861449373575226, 15584094305218387139, 1096358271829394613, 1529470393481296529, 362089145899586923, 4759023938040607680, 8045577689549541550, 1032320380402351332, 
            15010693726868240756, 10031381962380992446, 13616136326427811631, 1692165768332727562, 2108839775597975821, 17326020408477308405, 3206485344533219578, 18303973879305603268, 12814296373191120281, 
            10765987620087006141, 2742203196291721289, 13683262468274108609, 10134042530815520126, 17909690002769790095, 8637179986327094643, 12219733616878553351, 9227324760360540502, 3628395621830883812, 
            973398537799208969, 2967301329522480460, 5692930283327308413, 968899172769838687, 10327759251782185725, 12699618903397884819, 2704835705367505262, 15780812269047924098, 16361280639725058175, 
            12650951613230308876, 15562530674220209081, 11846704860053309561, 9982820104627732578, 2323442919270060514, 17906437972804021813, 6624016450053325651, 5088467518978949889, 5666511788609898502, 
            7314552822875358124, 170547778100806460, 5117368381706122026, 1145860449695299371, 3135944470876044986, 13967646923563520065, 5843985234160177859, 15009566086192926889, 18159122152459654418, 
            8737664961353780714]
    for i in range(len(ep["terminals"])):

        if ep["terminals"][i] == True:
            print(i)
    applied_seeds = []
    for i in range(len(seeds)):
        observation, info = env.reset(seed=seeds[i])

        threshold = 0.05
        interp_obs = interpolate_by_distance(observation, ep["observations"][0], threshold)  # Shape (N, obs_dim)

        new_obs = []
        new_actions = []
        new_dones = []
        for j in range(interp_obs.shape[0]-1):
            new_obs.append(observation)
            action = list(interp_obs[j+1][:7]) + [1.0]
            new_actions.append(action)
            new_dones.append(False)

            action[-1] = 255
            observation, reward, terminated, truncated, info = env.step(action)
            print

        save = input("Do you want to save?")
        if save == "Y":
            applied_seeds.append(seeds[i])
            new_obs = np.array(new_obs, np.float32)
            new_actions = np.array(new_actions, np.float32)
            new_dones = np.array(new_dones, bool)

            new_obs = np.concatenate([new_obs, rest_observations], axis=0)
            new_actions = np.concatenate([new_actions, rest_actions], axis=0)
            new_dones = np.concatenate([new_dones, rest_dones], axis=0)

            new_keys = ep["terminals"].shape[-1] - 1 + interp_obs.shape[0]  # the new episode starts at 0
            append_keys = [new_keys + k + 1 for k in keys]

            ep["observations"] = np.concatenate([ep["observations"], new_obs], axis = 0)
            ep["actions"] = np.concatenate([ep["actions"], new_actions], axis = 0)
            ep["terminals"] = np.concatenate([ep["terminals"], new_dones], axis = 0)
            
            ep["goal_idxs"] = np.concatenate([ep["goal_idxs"], np.array([new_keys]+append_keys)], axis = 0)

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

    print(applied_seeds) 
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('--env_module', type=str, default='sai', help='Environment (dataset) name.')
    parser.add_argument('--env_name', type=str, default='FrankaGolfCourseEnv-v0', help='Environment (dataset) name.')

    parser.add_argument('--file_name', type=str, default='filtered_data_20250604_140137_joint', help='Dataset file to load')

    args = parser.parse_args()

    main(args)