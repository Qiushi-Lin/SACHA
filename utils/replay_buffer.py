import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable


class ReplayBuffer:
    def __init__(self, config, num_agents):
        self.buffer_size = config['buffer_size']
        obs_shape = (config['K_obs'], 3, config['fov_size'][0], config['fov_size'][1])
        self.num_agents = num_agents
        self.obs_buff = np.zeros((self.buffer_size, self.num_agents, *obs_shape), dtype=float)
        self.action_buff = np.zeros((self.buffer_size, self.num_agents), dtype=float)
        self.reward_buff = np.zeros((self.buffer_size, self.num_agents), dtype=float)
        self.next_obs_buff = np.zeros((self.buffer_size, self.num_agents, *obs_shape), dtype=float)
        self.done_buff = np.zeros((self.buffer_size, self.num_agents), dtype=bool)
        self.info_buffer = [{} for _ in range(self.buffer_size)]
        self.hidden_buff = torch.zeros(self.buffer_size, self.num_agents, config['hidden_dim'], dtype=float)
        self.state_buff = np.zeros((self.buffer_size, self.num_agents, 2), dtype=float)
        self.ptr = 0
        self.num_samples = 0

    def __len__(self):
        return self.num_samples

    def add(self, b_obs, b_action, b_reward, b_next_obs, b_done, b_info, b_hidden, b_state):
        num_entries = len(b_obs)
        for i in range(num_entries):
            if self.ptr >= self.buffer_size:
                self.ptr = 0
            self.obs_buff[self.ptr] = b_obs[i]
            self.action_buff[self.ptr] = b_action[i]
            self.reward_buff[self.ptr] = b_reward[i]
            self.next_obs_buff[self.ptr] = b_next_obs[i]
            self.done_buff[self.ptr] = b_done[i]
            self.info_buffer[self.ptr] = b_info[i]
            self.hidden_buff[self.ptr] = b_hidden[i]
            self.state_buff[self.ptr] = b_state[i]
            self.ptr += 1
            self.num_samples = max(self.num_samples, self.ptr)

    def sample(self, sample_size, device):
        inds = np.random.choice(np.arange(self.num_samples, dtype=int), size=sample_size, replace=True)
        fn = lambda x: Variable(Tensor(x), requires_grad=False).to(device)
        obs = [fn(self.obs_buff[i]) for i in inds]
        action = [fn(self.action_buff[i]) for i in inds]
        reward = [fn(self.reward_buff[i]) for i in inds]
        next_obs = [fn(self.next_obs_buff[i]) for i in inds]
        done = [fn(self.done_buff[i]) for i in inds]
        info = [self.info_buffer[i] for i in inds]
        hidden = [fn(self.hidden_buff[i]) for i in inds]
        state = [fn(self.state_buff[i]) for i in inds]
        return obs, action, reward, next_obs, done, info, hidden, state
    
    def get_succes_rate(self):
        success_rate = 0.0
        for i in range(self.num_samples):
            success_rate += sum(self.done_buff[i])
        success_rate /= (self.num_agents * self.num_samples)
        return success_rate