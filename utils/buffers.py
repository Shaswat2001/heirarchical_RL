import numpy as np
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
        assert "observation" in data
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
    
class GCDataset:

    dataset: Dataset
    config: Any


