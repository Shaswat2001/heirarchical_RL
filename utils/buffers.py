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

        goals = [7, 16, 27, 36, 42, 56, 187, 754, 762, 771, 782, 791, 797, 811, 942, 1198, 1206, 1215, 1226, 1235, 1241, 1255, 1386, 1566, 1574, 1583, 1594, 1603, 1609, 1623, 1754, 1949, 1957, 1966, 1977, 1986, 1992, 2006, 2137, 2384, 2392, 2401, 2412, 2421, 2427, 2441, 2572, 2931, 2939, 2948, 2959, 2968, 2974, 2988, 3119, 3263, 3271, 3280, 3291, 3300, 3306, 3320, 3451, 3568, 3576, 3585, 3596, 3605, 3611, 3625, 3756, 3988, 3996, 4005, 4016, 4025, 4031, 4045, 4176, 4411, 4419, 4428, 4439, 4448, 4454, 4468, 4599, 4765, 4773, 4782, 4793, 4802, 4808, 4822, 4953, 5181, 5189, 5198, 5209, 5218, 5224, 5238, 5369, 5520, 5528, 5537, 5548, 5557, 5563, 5577, 5708, 5944, 5952, 5961, 5972, 5981, 5987, 6001, 6132, 6369, 6377, 6386, 6397, 6406, 6412, 6426, 6557, 6729, 6737, 6746, 6757, 6766, 6772, 6786, 6917, 7218, 7226, 7235, 7246, 7255, 7261, 7275, 7406, 7589, 7597, 7606, 7617, 7626, 7632, 7646, 7777, 7873, 7881, 7890, 7901, 7910, 7916, 7930, 8061, 8169, 8177, 8186, 8197, 8206, 8212, 8226, 8357, 8665, 8673, 8682, 8693, 8702, 8708, 8722, 8853, 9014, 9022, 9031, 9042, 9051, 9057, 9071, 9202, 9247, 9255, 9264, 9275, 9284, 9290, 9304, 9435, 9688, 9696, 9705, 9716, 9725, 9731, 9745, 9876, 10193, 10201, 10210, 10221, 10230, 10236, 10250, 10381, 10596, 10604, 10613, 10624, 10633, 10639, 10653, 10784, 10829, 10837, 10846, 10857, 10866, 10872, 10886, 11017, 11190, 11198, 11207, 11218, 11227, 11233, 11247, 11378, 11501, 11509, 11518, 11529, 11538, 11544, 11558, 11689, 11843, 11851, 11860, 11871, 11880, 11886, 11900, 12031, 12141, 12149, 12158, 12169, 12178, 12184, 12198, 12329, 12452, 12460, 12469, 12480, 12489, 12495, 12509, 12640, 12796, 12804, 12813, 12824, 12833, 12839, 12853, 12984, 13109, 13117, 13126, 13137, 13146, 13152, 13166, 13297, 13572, 13580, 13589, 13600, 13609, 13615, 13629, 13760, 13855, 13863, 13872, 13883, 13892, 13898, 13912, 14043, 14290, 14298, 14307, 14318, 14327, 14333, 14347, 14478, 14646, 14654, 14663, 14674, 14683, 14689, 14703, 14834, 15068, 15076, 15085, 15096, 15105, 15111, 15125, 15256, 15392, 15400, 15409, 15420, 15429, 15435, 15449, 15580, 15834, 15842, 15851, 15862, 15871, 15877, 15891, 16022, 16105, 16113, 16122, 16133, 16142, 16148, 16162, 16293, 16385, 16393, 16402, 16413, 16422, 16428, 16442, 16573, 16676, 16684, 16693, 16704, 16713, 16719, 16733, 16864, 17046, 17054, 17063, 17074, 17083, 17089, 17103, 17234, 17449, 17457, 17466, 17477, 17486, 17492, 17506, 17637, 17922, 17930, 17939, 17950, 17959, 17965, 17979, 18110, 18339, 18347, 18356, 18367, 18376, 18382, 18396, 18527, 18942, 18950, 18959, 18970, 18979, 18985, 18999, 19130, 19396, 19404, 19413, 19424, 19433, 19439, 19453, 19584, 19741, 19749, 19758, 19769, 19778, 19784, 19798, 19929, 20071, 20079, 20088, 20099, 20108, 20114, 20128, 20259, 20695, 20703, 20712, 20723, 20732, 20738, 20752, 20883, 21065, 21073, 21082, 21093, 21102, 21108, 21122, 21253, 21567, 21575, 21584, 21595, 21604, 21610, 21624, 21755, 21827, 21835, 21844, 21855, 21864, 21870, 21884, 22015, 22322, 22330, 22339, 22350, 22359, 22365, 22379, 22510, 22820, 22828, 22837, 22848, 22857, 22863, 22877, 23008, 23161, 23169, 23178, 23189, 23198, 23204, 23218, 23349, 23604, 23612, 23621, 23632, 23641, 23647, 23661, 23792, 23963, 23971, 23980, 23991, 24000, 24006, 24020, 24151, 24386, 24394, 24403, 24414, 24423, 24429, 24443, 24574, 24689, 24697, 24706, 24717, 24726, 24732, 24746, 24877, 25093, 25101, 25110, 25121, 25130, 25136, 25150, 25281, 25535, 25543, 25552, 25563, 25572, 25578, 25592, 25723, 25896, 25904, 25913, 25924, 25933, 25939, 25953, 26084, 26375, 26383, 26392, 26403, 26412, 26418, 26432, 26563, 26746, 26754, 26763, 26774, 26783, 26789, 26803, 26934, 27045, 27053, 27062, 27073, 27082, 27088, 27102, 27233, 27459, 27467, 27476, 27487, 27496, 27502, 27516, 27647, 27744, 27752, 27761, 27772, 27781, 27787, 27801, 27932, 28082, 28090, 28099, 28110, 28119, 28125, 28139, 28270, 28576, 28584, 28593, 28604, 28613, 28619, 28633, 28764, 28866, 28874, 28883, 28894, 28903, 28909, 28923, 29054, 29234, 29242, 29251, 29262, 29271, 29277, 29291, 29422, 29505, 29513, 29522, 29533, 29542, 29548, 29562, 29693, 29791, 29799, 29808, 29819, 29828, 29834, 29848, 29979, 30139, 30147, 30156, 30167, 30176, 30182, 30196, 30327, 30390, 30398, 30407, 30418, 30427, 30433, 30447, 30578, 30728, 30736, 30745, 30756, 30765, 30771, 30785, 30916, 31161, 31169, 31178, 31189, 31198, 31204, 31218, 31349, 31450, 31458, 31467, 31478, 31487, 31493, 31507, 31638, 31720, 31728, 31737, 31748, 31757, 31763, 31777, 31908, 32163, 32171, 32180, 32191, 32200, 32206, 32220, 32351, 32480, 32488, 32497, 32508, 32517, 32523, 32537, 32668, 32799, 32807, 32816, 32827, 32836, 32842, 32856, 32987, 33216, 33224, 33233, 33244, 33253, 33259, 33273, 33404]
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
        goals = [7, 16, 27, 36, 42, 56, 187, 754, 762, 771, 782, 791, 797, 811, 942, 1198, 1206, 1215, 1226, 1235, 1241, 1255, 1386, 1566, 1574, 1583, 1594, 1603, 1609, 1623, 1754, 1949, 1957, 1966, 1977, 1986, 1992, 2006, 2137, 2384, 2392, 2401, 2412, 2421, 2427, 2441, 2572, 2931, 2939, 2948, 2959, 2968, 2974, 2988, 3119, 3263, 3271, 3280, 3291, 3300, 3306, 3320, 3451, 3568, 3576, 3585, 3596, 3605, 3611, 3625, 3756, 3988, 3996, 4005, 4016, 4025, 4031, 4045, 4176, 4411, 4419, 4428, 4439, 4448, 4454, 4468, 4599, 4765, 4773, 4782, 4793, 4802, 4808, 4822, 4953, 5181, 5189, 5198, 5209, 5218, 5224, 5238, 5369, 5520, 5528, 5537, 5548, 5557, 5563, 5577, 5708, 5944, 5952, 5961, 5972, 5981, 5987, 6001, 6132, 6369, 6377, 6386, 6397, 6406, 6412, 6426, 6557, 6729, 6737, 6746, 6757, 6766, 6772, 6786, 6917, 7218, 7226, 7235, 7246, 7255, 7261, 7275, 7406, 7589, 7597, 7606, 7617, 7626, 7632, 7646, 7777, 7873, 7881, 7890, 7901, 7910, 7916, 7930, 8061, 8169, 8177, 8186, 8197, 8206, 8212, 8226, 8357, 8665, 8673, 8682, 8693, 8702, 8708, 8722, 8853, 9014, 9022, 9031, 9042, 9051, 9057, 9071, 9202, 9247, 9255, 9264, 9275, 9284, 9290, 9304, 9435, 9688, 9696, 9705, 9716, 9725, 9731, 9745, 9876, 10193, 10201, 10210, 10221, 10230, 10236, 10250, 10381, 10596, 10604, 10613, 10624, 10633, 10639, 10653, 10784, 10829, 10837, 10846, 10857, 10866, 10872, 10886, 11017, 11190, 11198, 11207, 11218, 11227, 11233, 11247, 11378, 11501, 11509, 11518, 11529, 11538, 11544, 11558, 11689, 11843, 11851, 11860, 11871, 11880, 11886, 11900, 12031, 12141, 12149, 12158, 12169, 12178, 12184, 12198, 12329, 12452, 12460, 12469, 12480, 12489, 12495, 12509, 12640, 12796, 12804, 12813, 12824, 12833, 12839, 12853, 12984, 13109, 13117, 13126, 13137, 13146, 13152, 13166, 13297, 13572, 13580, 13589, 13600, 13609, 13615, 13629, 13760, 13855, 13863, 13872, 13883, 13892, 13898, 13912, 14043, 14290, 14298, 14307, 14318, 14327, 14333, 14347, 14478, 14646, 14654, 14663, 14674, 14683, 14689, 14703, 14834, 15068, 15076, 15085, 15096, 15105, 15111, 15125, 15256, 15392, 15400, 15409, 15420, 15429, 15435, 15449, 15580, 15834, 15842, 15851, 15862, 15871, 15877, 15891, 16022, 16105, 16113, 16122, 16133, 16142, 16148, 16162, 16293, 16385, 16393, 16402, 16413, 16422, 16428, 16442, 16573, 16676, 16684, 16693, 16704, 16713, 16719, 16733, 16864, 17046, 17054, 17063, 17074, 17083, 17089, 17103, 17234, 17449, 17457, 17466, 17477, 17486, 17492, 17506, 17637, 17922, 17930, 17939, 17950, 17959, 17965, 17979, 18110, 18339, 18347, 18356, 18367, 18376, 18382, 18396, 18527, 18942, 18950, 18959, 18970, 18979, 18985, 18999, 19130, 19396, 19404, 19413, 19424, 19433, 19439, 19453, 19584, 19741, 19749, 19758, 19769, 19778, 19784, 19798, 19929, 20071, 20079, 20088, 20099, 20108, 20114, 20128, 20259, 20695, 20703, 20712, 20723, 20732, 20738, 20752, 20883, 21065, 21073, 21082, 21093, 21102, 21108, 21122, 21253, 21567, 21575, 21584, 21595, 21604, 21610, 21624, 21755, 21827, 21835, 21844, 21855, 21864, 21870, 21884, 22015, 22322, 22330, 22339, 22350, 22359, 22365, 22379, 22510, 22820, 22828, 22837, 22848, 22857, 22863, 22877, 23008, 23161, 23169, 23178, 23189, 23198, 23204, 23218, 23349, 23604, 23612, 23621, 23632, 23641, 23647, 23661, 23792, 23963, 23971, 23980, 23991, 24000, 24006, 24020, 24151, 24386, 24394, 24403, 24414, 24423, 24429, 24443, 24574, 24689, 24697, 24706, 24717, 24726, 24732, 24746, 24877, 25093, 25101, 25110, 25121, 25130, 25136, 25150, 25281, 25535, 25543, 25552, 25563, 25572, 25578, 25592, 25723, 25896, 25904, 25913, 25924, 25933, 25939, 25953, 26084, 26375, 26383, 26392, 26403, 26412, 26418, 26432, 26563, 26746, 26754, 26763, 26774, 26783, 26789, 26803, 26934, 27045, 27053, 27062, 27073, 27082, 27088, 27102, 27233, 27459, 27467, 27476, 27487, 27496, 27502, 27516, 27647, 27744, 27752, 27761, 27772, 27781, 27787, 27801, 27932, 28082, 28090, 28099, 28110, 28119, 28125, 28139, 28270, 28576, 28584, 28593, 28604, 28613, 28619, 28633, 28764, 28866, 28874, 28883, 28894, 28903, 28909, 28923, 29054, 29234, 29242, 29251, 29262, 29271, 29277, 29291, 29422, 29505, 29513, 29522, 29533, 29542, 29548, 29562, 29693, 29791, 29799, 29808, 29819, 29828, 29834, 29848, 29979, 30139, 30147, 30156, 30167, 30176, 30182, 30196, 30327, 30390, 30398, 30407, 30418, 30427, 30433, 30447, 30578, 30728, 30736, 30745, 30756, 30765, 30771, 30785, 30916, 31161, 31169, 31178, 31189, 31198, 31204, 31218, 31349, 31450, 31458, 31467, 31478, 31487, 31493, 31507, 31638, 31720, 31728, 31737, 31748, 31757, 31763, 31777, 31908, 32163, 32171, 32180, 32191, 32200, 32206, 32220, 32351, 32480, 32488, 32497, 32508, 32517, 32523, 32537, 32668, 32799, 32807, 32816, 32827, 32836, 32842, 32856, 32987, 33216, 33224, 33233, 33244, 33253, 33259, 33273, 33404]
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
