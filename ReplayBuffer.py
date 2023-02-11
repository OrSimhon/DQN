import random
from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def store(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)

        states = np.array([elem[0] for elem in mini_batch])
        actions = np.array([elem[1] for elem in mini_batch])
        next_states = np.array([elem[2] for elem in mini_batch])
        rewards = np.array([elem[3] for elem in mini_batch], dtype=np.float32)
        not_dones = np.array([elem[4] for elem in mini_batch])

        return states, actions, next_states, rewards, not_dones

    def __len__(self):
        return len(self.memory)
