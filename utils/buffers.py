import numpy as np
import dataclasses
from typing import Any

import jax 
from flax.core.frozen_dict import FrozenDict

def get_dataset_size(data):
    leaf_size = jax.tree.map(lambda arr: len(arr), data) # similar to map function in python but for Pytrees
    return max(jax.tree.leaves(leaf_size))

class Dataset(FrozenDict):

    @classmethod
    def create(cls, freeze=True, **fields):

        data = fields
        assert "observations" in data
        if freeze:
            jax.tree.map(lambda field: field.setflags(write=False), data) # sets the array to become immutable

        return cls(data)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.size = get_dataset_size(self._dict)

    def get_random_idxs(self, batch_size):

        return np.random.randint(self.size, size=batch_size)
    
    def create_biased_data(self, goals):

        goal_rows = [[7, 16, 27, 36, 42, 56, 187],[198, 206, 221, 352]]
        goals = goals[11:]
        goal_rows = [goals[i:i+8] for i in range(0, len(goals), 8)]
        biased_ranges = []

        for i, row in enumerate(goal_rows):
            if len(row) < 8:
                continue  # Skip incomplete row

            # Add ranges between row[1] and row[7]
            for j in range(0, 6):
                biased_ranges.append((row[j], row[j+1]))

            # # Add boundary to next row if exists
            # if i + 1 < len(goal_rows):
            #     next_row = goal_rows[i + 1]
            #     if len(next_row) >= 1:
            #         biased_ranges.append((row[7], next_row[0]))

        # Flatten all allowed indices in biased regions
        biased_idxs = []
        for start, end in biased_ranges:
            biased_idxs.extend(list(range(start, min(end, self.size))))  # Clip to dataset size

        # Deduplicate and clip
        self.biased_idxs = list(set([i for i in biased_idxs if i < self.size]))
    
    def get_biased_idxs(self, batch_size, proportion=1.0):
        """
        Samples batch with emphasis on intermediate regions between goals[1:7] in each row of 8,
        and boundary between last of row N and first of row N+1.
        
        Args:
            batch_size: Total number of samples.
            goals: List of goal indices.
            proportion: Fraction of samples to draw from the biased region.
        """
        # Group goals into rows of 8

        # Sample from biased region
        n_biased = int(batch_size * proportion)
        n_random = batch_size - n_biased

        biased_sample_idxs = np.random.choice(self.biased_idxs, size=n_biased, replace=len(self.biased_idxs) < n_biased)
        random_sample_idxs = self.get_random_idxs(n_random)

        all_idxs = np.concatenate([biased_sample_idxs, random_sample_idxs])
        return all_idxs

    
    def biased_sample(self, idxs = None):

        return self.get_subset(idxs)
    
    def sample(self, batch_size, idxs = None):

        if idxs is None:
            idxs = self.get_random_idxs(batch_size)

        return self.get_subset(idxs)

    def get_subset(self, idxs):

        result = jax.tree.map(lambda arr: arr[idxs], self._dict)
        if 'next_observations' not in result:
            result['next_observations'] = self._dict['observations'][np.minimum(idxs + 1, self.size - 1)]
        return result
    
class HindsightReplayBuffer(Dataset):

    pass

class ReplayBuffer(Dataset):

    pass
    
@dataclasses.dataclass
class GCDataset:

    dataset: Dataset
    config: Any

    def __post_init__(self):

        self.size = self.dataset.size

        (self.where_terminate, ) = np.nonzero(self.dataset["terminals"] > 0)
        goals = [7, 16, 27, 36, 42, 56, 187, 198, 206, 221, 352, 
                 468, 475, 484, 495, 504, 510, 524, 655, 
                 779, 786, 795, 806, 815, 821, 835, 966, 
                 1131, 1138, 1147, 1158, 1167, 1173, 1187, 1318, 
                 1433, 1440, 1449, 1460, 1469, 1475, 1489, 1620, 
                 1686, 1693, 1702, 1713, 1722, 1728, 1742, 1873, 
                 2261, 2268, 2277, 2288, 2297, 2303, 2317, 2448,
                    2516, 2523, 2532, 2543, 2552, 2558, 2572, 2703,
                    3016, 3023, 3032, 3043, 3052, 3058, 3072, 3203,
                    3327, 3334, 3343, 3354, 3363, 3369, 3383, 3514,
                    3574, 3581, 3590, 3601, 3610, 3616, 3630, 3761,
                    3994, 4001, 4010, 4021, 4030, 4036, 4050, 4181,
                    4436, 4443, 4452, 4463, 4472, 4478, 4492, 4623,
                    4787, 4794, 4803, 4814, 4823, 4829, 4843, 4974,
                    5206, 5213, 5222, 5233, 5242, 5248, 5262, 5393,
                    5587, 5594, 5603, 5614, 5623, 5629, 5643, 5774,
                    5892, 5899, 5908, 5919, 5928, 5934, 5948, 6079,
                    6471, 6478, 6487, 6498, 6507, 6513, 6527, 6658,
                    6998, 7005, 7014, 7025, 7034, 7040, 7054, 7185,
                    7390, 7397, 7406, 7417, 7426, 7432, 7446, 7577,
                    7665, 7672, 7681, 7692, 7701, 7707, 7721, 7852,
                    7998, 8005, 8014, 8025, 8034, 8040, 8054, 8185,
                    8355, 8362, 8371, 8382, 8391, 8397, 8411, 8542,
                    8648, 8655, 8664, 8675, 8684, 8690, 8704, 8835,
                    9050, 9057, 9066, 9077, 9086, 9092, 9106, 9237,
                    9696, 9703, 9712, 9723, 9732, 9738, 9752, 9883,
                    10039, 10046, 10055, 10066, 10075, 10081, 10095, 10226,
                    10310, 10317, 10326, 10337, 10346, 10352, 10366, 10497,
                    10753, 10760, 10769, 10780, 10789, 10795, 10809, 10940,
                    11095, 11102, 11111, 11122, 11131, 11137, 11151, 11282,
                    11517, 11524, 11533, 11544, 11553, 11559, 11573, 11704,
                    11894, 11901, 11910, 11921, 11930, 11936, 11950, 12081,
                    12192, 12199, 12208, 12219, 12228, 12234, 12248, 12379,
                    12454, 12461, 12470, 12481, 12490, 12496, 12510, 12641,
                    12764, 12771, 12780, 12791, 12800, 12806, 12820, 12951,
                    13234, 13241, 13250, 13261, 13270, 13276, 13290, 13421,
                    13503, 13510, 13519, 13530, 13539, 13545, 13559, 13690,
                    13868, 13875, 13884, 13895, 13904, 13910, 13924, 14055,
                    14212, 14219, 14228, 14239, 14248, 14254, 14268, 14399,
                    14578, 14585, 14594, 14605, 14614, 14620, 14634, 14765,
                    14912, 14919, 14928, 14939, 14948, 14954, 14968, 15099,
                    15362, 15369, 15378, 15389, 15398, 15404, 15418, 15549,
                    15679, 15686, 15695, 15706, 15715, 15721, 15735, 15866,
                    16040, 16047, 16056, 16067, 16076, 16082, 16096, 16227,
                    16418, 16425, 16434, 16445, 16454, 16460, 16474, 16605,
                    16893, 16900, 16909, 16920, 16929, 16935, 16949, 17080,
                    17159, 17166, 17175, 17186, 17195, 17201, 17215, 17346,
                    17618, 17625, 17634, 17645, 17654, 17660, 17674, 17805,
                    18208, 18215, 18224, 18235, 18244, 18250, 18264, 18395,
                    18550, 18557, 18566, 18577, 18586, 18592, 18606, 18737,
                    18866, 18873, 18882, 18893, 18902, 18908, 18922, 19053,
                    19174, 19181, 19190, 19201, 19210, 19216, 19230, 19361,
                    19494, 19501, 19510, 19521, 19530, 19536, 19550, 19681,
                    19806, 19813, 19822, 19833, 19842, 19848, 19862, 19993,
                    20095, 20102, 20111, 20122, 20131, 20137, 20151, 20282,
                    20517, 20524, 20533, 20544, 20553, 20559, 20573, 20704,
                    20830, 20837, 20846, 20857, 20866, 20872, 20886, 21017,
                    21209, 21216, 21225, 21236, 21245, 21251, 21265, 21396,
                    21474, 21481, 21490, 21501, 21510, 21516, 21530, 21661,
                    21903, 21910, 21919, 21930, 21939, 21945, 21959, 22090,
                    22257, 22264, 22273, 22284, 22293, 22299, 22313, 22444,
                    22676, 22683, 22692, 22703, 22712, 22718, 22732, 22863,
                    22981, 22988, 22997, 23008, 23017, 23023, 23037, 23168,
                    23236, 23243, 23252, 23263, 23272, 23278, 23292, 23423,
                    23655, 23662, 23671, 23682, 23691, 23697, 23711, 23842,
                    24071, 24078, 24087, 24098, 24107, 24113, 24127, 24258,
                    24475, 24482, 24491, 24502, 24511, 24517, 24531, 24662,
                    24834, 24841, 24850, 24861, 24870, 24876, 24890, 25021,
                    25245, 25252, 25261, 25272, 25281, 25287, 25301, 25432,
                    25611, 25618, 25627, 25638, 25647, 25653, 25667, 25797]
        self.boundaries = np.array(goals)#188, 198, 206, 221, len(self.dataset['observations'])-1])
        
        self.dataset.create_biased_data(self.boundaries)
        assert self.where_terminate[-1] == self.size - 1

        assert np.isclose(
            self.config['value_p_curgoal'] + self.config['value_p_trajgoal'] + self.config['value_p_randomgoal'], 1.0
        )
        assert np.isclose(
            self.config['actor_p_curgoal'] + self.config['actor_p_trajgoal'] + self.config['actor_p_randomgoal'], 1.0
        )

    def sample(self, batch_size, idxs = None, sample="biased"):

        if sample=="biased":
            idxs = self.dataset.get_biased_idxs(batch_size, proportion=0.75)
            
            batch = self.dataset.biased_sample(idxs)
        else:
            if idxs is None:
                idxs = self.dataset.get_random_idxs(batch_size)
            
            batch = self.dataset.sample(batch_size, idxs)

        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        actor_goal_idxs = self.sample_goals(
            idxs,
            self.config['actor_p_curgoal'],
            self.config['actor_p_trajgoal'],
            self.config['actor_p_randomgoal'],
            self.config['actor_geom_sample'],
        )

        batch['value_goals'] = self.get_observations(value_goal_idxs)
        batch['actor_goals'] = self.get_observations(actor_goal_idxs)
        successes = (idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        return batch

    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample):
        """Sample goals for the given indices."""
        batch_size = len(idxs)
        
        # Random goals.
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)
        search_idxs = np.searchsorted(self.boundaries, idxs)
        search_idxs = search_idxs = np.clip(search_idxs, 0, len(self.boundaries) - 1)
        cap_idxs = self.boundaries[search_idxs]
        # print("cap: ", idxs)

        # Goals from the same trajectory (excluding the current state, unless it is the final state).
        final_state_idxs = self.where_terminate[np.searchsorted(self.where_terminate, idxs)]
        if geom_sample:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)  # in [1, inf)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
            traj_goal_idxs = np.minimum(traj_goal_idxs, cap_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
            traj_goal_idxs = np.minimum(traj_goal_idxs, cap_idxs)
            # traj_goal_idxs = np.round(np.minimum(idxs + 4, final_state_idxs))
        if p_curgoal == 1.0:
            goal_idxs = idxs
        else:

            goal_idxs = np.where(
                np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal), traj_goal_idxs, random_goal_idxs
            )

            # Goals at the current state.
            goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs)
        
        goal_idxs = cap_idxs
        # print("idxs: ", idxs)
        # print("goal idxs: ", goal_idxs)
        return goal_idxs
    
    def get_observations(self, idxs):
        return jax.tree.map(lambda arr: arr[idxs], self.dataset['observations'])

@dataclasses.dataclass
class HGCDataset(GCDataset):

    def sample(self, batch_size, idxs = None):

        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)
        
        batch = self.dataset.sample(batch_size, idxs)
        goals = [7, 16, 27, 36, 42, 56, 187, 468, 476, 485, 496, 505, 511, 525, 656, 779, 787, 796, 807, 816, 822, 836, 967, 1131, 1139, 1148, 1159, 1168, 1174, 1188, 1319, 1433, 1441, 1450, 1461, 1470, 1476, 1490, 1621, 1686, 1694, 1703, 1714, 1723, 1729, 1743, 1874, 2261, 2269, 2278, 2289, 2298, 2304, 2318, 2449, 2516, 2524, 2533, 2544, 2553, 2559, 2573, 2704, 3016, 3024, 3033, 3044, 3053, 3059, 3073, 3204, 3327, 3335, 3344, 3355, 3364, 3370, 3384, 3515, 3574, 3582, 3591, 3602, 3611, 3617, 3631, 3762, 3994, 4002, 4011, 4022, 4031, 4037, 4051, 4182, 4436, 4444, 4453, 4464, 4473, 4479, 4493, 4624, 4787, 4795, 4804, 4815, 4824, 4830, 4844, 4975, 5206, 5214, 5223, 5234, 5243, 5249, 5263, 5394, 5587, 5595, 5604, 5615, 5624, 5630, 5644, 5775, 5892, 5900, 5909, 5920, 5929, 5935, 5949, 6080, 6471, 6479, 6488, 6499, 6508, 6514, 6528, 6659, 6998, 7006, 7015, 7026, 7035, 7041, 7055, 7186, 7390, 7398, 7407, 7418, 7427, 7433, 7447, 7578, 7665, 7673, 7682, 7693, 7702, 7708, 7722, 7853, 7998, 8006, 8015, 8026, 8035, 8041, 8055, 8186, 8355, 8363, 8372, 8383, 8392, 8398, 8412, 8543, 8648, 8656, 8665, 8676, 8685, 8691, 8705, 8836, 9050, 9058, 9067, 9078, 9087, 9093, 9107, 9238, 9696, 9704, 9713, 9724, 9733, 9739, 9753, 9884, 10039, 10047, 10056, 10067, 10076, 10082, 10096, 10227, 10310, 10318, 10327, 10338, 10347, 10353, 10367, 10498, 10753, 10761, 10770, 10781, 10790, 10796, 10810, 10941, 11095, 11103, 11112, 11123, 11132, 11138, 11152, 11283, 11517, 11525, 11534, 11545, 11554, 11560, 11574, 11705, 11894, 11902, 11911, 11922, 11931, 11937, 11951, 12082, 12192, 12200, 12209, 12220, 12229, 12235, 12249, 12380, 12454, 12462, 12471, 12482, 12491, 12497, 12511, 12642, 12764, 12772, 12781, 12792, 12801, 12807, 12821, 12952, 13234, 13242, 13251, 13262, 13271, 13277, 13291, 13422, 13503, 13511, 13520, 13531, 13540, 13546, 13560, 13691, 13868, 13876, 13885, 13896, 13905, 13911, 13925, 14056, 14212, 14220, 14229, 14240, 14249, 14255, 14269, 14400, 14578, 14586, 14595, 14606, 14615, 14621, 14635, 14766, 14912, 14920, 14929, 14940, 14949, 14955, 14969, 15100, 15362, 15370, 15379, 15390, 15399, 15405, 15419, 15550, 15679, 15687, 15696, 15707, 15716, 15722, 15736, 15867, 16040, 16048, 16057, 16068, 16077, 16083, 16097, 16228, 16418, 16426, 16435, 16446, 16455, 16461, 16475, 16606, 16893, 16901, 16910, 16921, 16930, 16936, 16950, 17081, 17159, 17167, 17176, 17187, 17196, 17202, 17216, 17347, 17618, 17626, 17635, 17646, 17655, 17661, 17675, 17806, 18208, 18216, 18225, 18236, 18245, 18251, 18265, 18396, 18550, 18558, 18567, 18578, 18587, 18593, 18607, 18738, 18866, 18874, 18883, 18894, 18903, 18909, 18923, 19054, 19174, 19182, 19191, 19202, 19211, 19217, 19231, 19362, 19494, 19502, 19511, 19522, 19531, 19537, 19551, 19682, 19806, 19814, 19823, 19834, 19843, 19849, 19863, 19994, 20095, 20103, 20112, 20123, 20132, 20138, 20152, 20283, 20517, 20525, 20534, 20545, 20554, 20560, 20574, 20705, 20830, 20838, 20847, 20858, 20867, 20873, 20887, 21018, 21209, 21217, 21226, 21237, 21246, 21252, 21266, 21397, 21474, 21482, 21491, 21502, 21511, 21517, 21531, 21662, 21903, 21911, 21920, 21931, 21940, 21946, 21960, 22091, 22257, 22265, 22274, 22285, 22294, 22300, 22314, 22445, 22676, 22684, 22693, 22704, 22713, 22719, 22733, 22864, 22981, 22989, 22998, 23009, 23018, 23024, 23038, 23169, 23236, 23244, 23253, 23264, 23273, 23279, 23293, 23424, 23655, 23663, 23672, 23683, 23692, 23698, 23712, 23843, 24071, 24079, 24088, 24099, 24108, 24114, 24128, 24259, 24475, 24483, 24492, 24503, 24512, 24518, 24532, 24663, 24834, 24842, 24851, 24862, 24871, 24877, 24891, 25022, 25245, 25253, 25262, 25273, 25282, 25288, 25302, 25433, 25611, 25619, 25628, 25639, 25648, 25654, 25668, 25799]
        self.boundaries = np.array(goals)#188, 198, 206, 221, len(self.dataset['observations'])-1])

        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        batch['value_goals'] = self.get_observations(value_goal_idxs)
        successes = (idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        final_state_idxs = self.where_terminate[np.searchsorted(self.where_terminate, idxs)]
        low_level_subgoal_idxs = np.minimum(idxs + self.config["subgoal_step"], final_state_idxs)
        batch["low_level_actor_goals"] = self.get_observations(low_level_subgoal_idxs)

        if self.config['actor_geom_sample']:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)  # in [1, inf)
            high_level_traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            high_level_traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)

        high_level_target_idxs = np.minimum(idxs + self.config["subgoal_step"], high_level_traj_goal_idxs)

        high_level_random_goals_idxs = self.dataset.get_random_idxs(batch_size)
        high_level_random_targets_idxs = np.minimum(idxs + self.config["subgoal_step"], final_state_idxs)

        pick_random = np.random.rand(batch_size) < self.config['actor_p_randomgoal']
        high_goal_idxs = np.where(pick_random, high_level_random_goals_idxs, high_level_traj_goal_idxs)
        high_target_idxs = np.where(pick_random, high_level_random_targets_idxs, high_level_target_idxs)
        batch['high_actor_goals'] = self.get_observations(high_goal_idxs)
        batch['high_actor_targets'] = self.get_observations(high_target_idxs)

        return batch

    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample):
        """Sample goals for the given indices."""
        batch_size = len(idxs)

        # Random goals.
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)
        cap_idxs = self.boundaries[np.searchsorted(self.boundaries, idxs)]
        # Goals from the same trajectory (excluding the current state, unless it is the final state).
        final_state_idxs = self.where_terminate[np.searchsorted(self.where_terminate, idxs)]
        if geom_sample:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)  # in [1, inf)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        if p_curgoal == 1.0:
            goal_idxs = idxs
        else:
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal), traj_goal_idxs, random_goal_idxs
            )

            # Goals at the current state.
            goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs)

        return cap_idxs
    
    def get_observations(self, idxs):
        return jax.tree.map(lambda arr: arr[idxs], self.dataset['observations'])

buffers = {
    "GCDataset" : GCDataset,
    "HGCDataset" : HGCDataset
}
