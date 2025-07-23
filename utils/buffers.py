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
    
@dataclasses.dataclass
class GCDataset:

    dataset: Dataset
    config: Any

    def __post_init__(self):

        self.size = self.dataset.size

        (self.where_terminate, ) = np.nonzero(self.dataset["terminals"] > 0)
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

        if p_curgoal == 1.0:
            goal_idxs = idxs
        else:

            goal_idxs = np.where(
                np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal), traj_goal_idxs, random_goal_idxs
            )

            # Goals at the current state.
            goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs)
        
        return goal_idxs
    
    def get_observations(self, idxs):
        return jax.tree.map(lambda arr: arr[idxs], self.dataset['observations'])

@dataclasses.dataclass
class HGCDataset(GCDataset):

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

        return goal_idxs
    
    def get_observations(self, idxs):
        return jax.tree.map(lambda arr: arr[idxs], self.dataset['observations'])

buffers = {
    "GCDataset" : GCDataset,
    "HGCDataset" : HGCDataset
}
