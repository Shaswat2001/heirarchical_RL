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
        goals = [7, 16, 27, 36, 42, 56, 187, 468, 476, 485, 496, 505, 511, 525, 656, 779, 787, 796, 807, 816, 822, 836, 967, 1131, 1139, 1148, 1159, 1168, 1174, 1188, 1319, 1433, 1441, 1450, 1461, 1470, 1476, 1490, 1621, 1686, 1694, 1703, 1714, 1723, 1729, 1743, 1874, 2261, 2269, 2278, 2289, 2298, 2304, 2318, 2449, 2516, 2524, 2533, 2544, 2553, 2559, 2573, 2704, 3016, 3024, 3033, 3044, 3053, 3059, 3073, 3204, 3327, 3335, 3344, 3355, 3364, 3370, 3384, 3515, 3574, 3582, 3591, 3602, 3611, 3617, 3631, 3762, 3994, 4002, 4011, 4022, 4031, 4037, 4051, 4182, 4436, 4444, 4453, 4464, 4473, 4479, 4493, 4624, 4787, 4795, 4804, 4815, 4824, 4830, 4844, 4975, 5206, 5214, 5223, 5234, 5243, 5249, 5263, 5394, 
                 5587, 5595, 5604, 5615, 5624, 5630, 5644, 5775, 5892, 5900, 5909, 5920, 5929, 5935, 5949, 6080, 6471, 6479, 6488, 6499, 6508, 6514, 6528, 6659, 6998, 7006, 7015, 7026, 7035, 7041, 7055, 7186, 7390, 7398, 7407, 7418, 7427, 7433, 7447, 7578, 7665, 7673, 7682, 7693, 7702, 7708, 7722, 7853, 7998, 8006, 8015, 8026, 8035, 8041, 8055, 8186, 8355, 8363, 8372, 8383, 8392, 8398, 8412, 8543, 8648, 8656, 8665, 8676, 8685, 8691, 8705, 8836, 9050, 9058, 9067, 9078, 9087, 9093, 9107, 9238, 9696, 9704, 9713, 9724, 9733, 9739, 9753, 9884, 10039, 10047, 10056, 10067, 10076, 10082, 10096, 10227, 10310, 10318, 10327, 10338, 10347, 10353, 10367, 10498, 10753, 10761, 10770, 10781, 10790, 10796, 10810, 10941, 11095, 11103, 11112, 11123, 11132, 11138, 11152, 11283, 11517, 11525, 11534, 11545, 11554, 11560, 11574, 11705, 11894, 11902, 11911, 11922, 11931, 11937, 11951, 
                 12082, 12192, 12200, 12209, 12220, 12229, 12235, 12249, 12380, 12454, 12462, 12471, 12482, 12491, 12497, 12511, 12642, 12764, 12772, 12781, 12792, 12801, 12807, 12821, 12952, 13234, 13242, 13251, 13262, 13271, 13277, 13291, 13422, 13503, 13511, 13520, 13531, 13540, 13546, 13560, 13691, 13868, 13876, 13885, 13896, 13905, 13911, 13925, 14056, 14212, 14220, 14229, 14240, 14249, 14255, 14269, 14400, 14578, 14586, 14595, 14606, 14615, 14621, 14635, 14766, 14912, 14920, 14929, 14940, 14949, 14955, 14969, 15100, 15362, 15370, 15379, 15390, 15399, 15405, 15419, 15550, 15679, 15687, 15696, 15707, 15716, 15722, 15736, 15867, 16040, 16048, 16057, 16068, 16077, 16083, 16097, 16228, 16418, 16426, 16435, 16446, 16455, 16461, 16475, 16606, 16893, 16901, 16910, 16921, 16930, 16936, 16950, 17081, 17159, 17167, 17176, 17187, 17196, 17202, 17216, 17347, 17618, 17626, 17635, 17646, 17655, 17661, 17675, 17806, 18208, 18216, 18225, 
                 18236, 18245, 18251, 18265, 18396, 18550, 18558, 18567, 18578, 18587, 18593, 18607, 18738, 18866, 18874, 18883, 18894, 18903, 18909, 18923, 19054, 19174, 19182, 19191, 19202, 19211, 19217, 19231, 19362, 19494, 19502, 19511, 19522, 19531, 19537, 19551, 19682, 19806, 19814, 19823, 19834, 19843, 19849, 19863, 19994, 20095, 20103, 20112, 20123, 20132, 20138, 20152, 20283, 20517, 20525, 20534, 20545, 20554, 20560, 20574, 20705, 20830, 20838, 20847, 20858, 20867, 20873, 20887, 21018, 21209, 21217, 21226, 21237, 21246, 21252, 21266, 21397, 21474, 21482, 21491, 21502, 21511, 21517, 21531, 21662, 21903, 21911, 21920, 21931, 21940, 21946, 21960, 22091, 22257, 22265, 22274, 22285, 22294, 22300, 22314, 22445, 22676, 22684, 22693, 22704, 22713, 22719, 22733, 22864, 22981, 22989, 22998, 23009, 23018, 23024, 23038, 23169, 23236, 23244, 23253, 23264, 23273, 23279, 23293, 23424, 23655, 23663, 23672, 23683, 23692, 23698, 23712, 
                 23843, 24071, 24079, 24088, 24099, 24108, 24114, 24128, 24259, 24475, 24483, 24492, 24503, 24512, 24518, 24532, 24663, 24834, 24842, 24851, 24862, 24871, 24877, 24891, 25022, 25245, 25253, 25262, 25273, 25282, 25288, 25302, 25433, 25611, 25619, 25628, 25639, 25648, 25654, 25668, 25798]
        self.boundaries = np.array(goals)#188, 198, 206, 221, len(self.dataset['observations'])-1])

        assert self.where_terminate[-1] == self.size - 1

        assert np.isclose(
            self.config['value_p_curgoal'] + self.config['value_p_trajgoal'] + self.config['value_p_randomgoal'], 1.0
        )
        assert np.isclose(
            self.config['actor_p_curgoal'] + self.config['actor_p_trajgoal'] + self.config['actor_p_randomgoal'], 1.0
        )

    def sample(self, batch_size, idxs = None):

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
        cap_idxs = self.boundaries[np.searchsorted(self.boundaries, idxs)]
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
