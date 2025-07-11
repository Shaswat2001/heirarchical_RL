import collections
import os
import platform
import time

import gymnasium
import numpy as np
from gymnasium.spaces import Box

import ogbench
import sai_mujoco
from utils.buffers import Dataset


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if 'goal' in info:
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self.get_observation(), reward, terminated, truncated, info


def make_env_and_datasets(dataset_name, frame_stack=None):
    """Make OGBench environment and datasets.

    Args:
        dataset_name: Name of the dataset.
        frame_stack: Number of frames to stack.

    Returns:
        A tuple of the environment, training dataset, and validation dataset.
    """
    # Use compact dataset to save memory.
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(dataset_name, compact_dataset=True)
    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)

    env.reset()

    return env, train_dataset, val_dataset

def make_sai_datasets(env_name):

    dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    env = gymnasium.make(env_name, keyframe="init_frame")

    with np.load(f'{dir_name}/dataset/{env_name}/train/FrankaGolfCourseEnv-v0_train_augmented_new.npz', allow_pickle=True) as data:
        train_data = {key: data[key] for key in data}

    # with np.load(f'{dir_name}/dataset/{env_name}/val/{env_name}_val.npz', allow_pickle=True) as data:
    #     val_data = {key: data[key] for key in data}
    

    # scaler = MinMaxScaler(feature_range=(-1, 1))

    # # Fit and transform
    # train_data["actions"] = scaler.fit_transform(train_data["actions"])
    # val_data["actions"] = scaler.transform(val_data["actions"])
    # train_data["actions"] = 2.*(train_data["actions"] - np.min(train_data["actions"]))/np.ptp(train_data["actions"])-1
    # val_data["actions"] = 2.*(val_data["actions"] - np.min(val_data["actions"]))/np.ptp(val_data["actions"])-1
    # train_data["actions"][:,:-1] = train_data["actions"][:,:-1] * 100
    del train_data["goal_idxs"]
    from sklearn.preprocessing import MinMaxScaler
    import pickle

    scaler = MinMaxScaler(feature_range=(-1,1))
    train_data["actions"] = scaler.fit_transform(train_data["actions"])
    # print(train_data["actions"].shape)
    # print(train_data['actions'])
    with open('standard_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print(train_data["actions"].shape)
    # val_data["actions"][:,:-1] = val_data["actions"][:,:-1] * 100
    # for i in range(len(train_data["actions"])):
    
    #     print(f'{i} : {train_data["actions"][i]}')
    train_dataset = Dataset.create(**train_data)
    # val_dataset = Dataset.create(**val_data)

    return env, train_dataset, None




