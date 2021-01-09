import numpy as np
import torch
import torch.nn as nn
import kornia

class ReplayBuffer:
    def __init__(self, state_shape, action_size, capacity, seed, image_pad, device):
        np.random.seed(seed=seed)
        self.seed = torch.manual_seed(seed)
        self.state_size = state_shape
        self.action_size = action_size
        self.capacity = capacity
        self.device = device
        self.states = np.empty((capacity, *state_shape), dtype=np.uint8)
        self.next_states = np.empty((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, *action_size), dtype=np.int8)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.int8)
        self.aug_trans = nn.Sequential(
            nn.ReplicationPad2d(image_pad),
            kornia.augmentation.RandomCrop((state_shape[-1], state_shape[-1])))
        self.idx = 0
        self.full = False

    def add(self, state, reward, action, next_state, done):
        np.copyto(self.states[self.idx], state)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_states[self.idx], next_state)
        np.copyto(self.dones[self.idx],  done)
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, batch_size):
        ids = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)
        states = torch.as_tensor(self.states[ids], device=self.device, dtype=torch.float32)
        next_states = torch.as_tensor(self.next_states[ids], device=self.device, dtype=torch.float32)
        actions = torch.as_tensor(self.actions[ids], device=self.device, dtype=torch.int64)
        rewards = torch.as_tensor(self.rewards[ids], device=self.device)
        dones = torch.as_tensor(self.dones[ids], device=self.device)
        states = self.aug_trans(states)
        next_states = self.aug_trans(next_states)
        return states, rewards, actions, next_states, dones
    
    def save_memory(self, filename):
        """
        Use numpy save function to store the data in a given file
        """
        with open(filename + '/obses.npy', 'wb') as f:
            np.save(f, self.obses)

        with open(filename + '/actions.npy', 'wb') as f:
            np.save(f, self.actions)

        with open(filename + '/next_obses.npy', 'wb') as f:
            np.save(f, self.next_obses)

        with open(filename + '/rewards.npy', 'wb') as f:
            np.save(f, self.rewards)

        with open(filename + '/not_dones.npy', 'wb') as f:
            np.save(f, self.not_dones)

        with open(filename + '/not_dones_no_max.npy', 'wb') as f:
            np.save(f, self.not_dones_no_max)

        with open(filename + '/index.txt', 'w') as f:
            f.write("{}".format(self.idx))
        print("save buffer to {}".format(filename))

    def load_memory(self, filename):
        """
        Use numpy load function to store the data in a given file
        """
        with open(filename + '/obses.npy', 'rb') as f:
            self.obses = np.load(f)

        with open(filename + '/actions.npy', 'rb') as f:
            self.actions = np.load(f)

        with open(filename + '/next_obses.npy', 'rb') as f:
            self.next_obses = np.load(f)

        with open(filename + '/index.txt', 'r') as f:
            self.idx = int(f.read())

