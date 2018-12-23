from collections import namedtuple, deque
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
        """
        random.seed(seed)
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.experience = namedtuple(
            "Experience",
            field_names=["obs", "obs_full", "action", "reward", "next_state", "next_state_full", "done"])
        
    def add(self, obs, obs_full, action, reward, next_state, next_state_full, done):
        """Add a new experience to memory."""

        e = self.experience(obs, obs_full, action, reward, next_state, next_state_full, done)
        self.memory.append(e)

    def sample(self, n):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=n)

        obs = np.array([e.obs for e in experiences if e is not None])
        obs_full = np.array([e.obs_full for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None])
        rewards = np.array([e.reward for e in experiences if e is not None])
        next_state = np.array([e.next_state for e in experiences if e is not None])
        next_state_full = np.array([e.next_state_full for e in experiences if e is not None])
        dones = np.array([e.done for e in experiences if e is not None])

        return obs, obs_full, actions, rewards, next_state, next_state_full, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def size(self):
        return len(self.memory)