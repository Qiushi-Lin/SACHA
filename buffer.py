import numpy as np
import time
from copy import deepcopy
from typing import Tuple
import threading
import ray
import torch
import numpy as np

# config
import yaml
config = yaml.safe_load(open("./config.yaml", 'r'))


class SumTree:
    '''used for prioritized experience replay''' 
    def __init__(self, capacity: int):
        layer = 1
        while 2**(layer-1) < capacity:
            layer += 1
        assert 2**(layer-1) == capacity, 'capacity only allow n**2 size'
        self.layer = layer
        self.tree = np.zeros(2**layer-1, dtype=np.float64)
        self.capacity = capacity
        self.size = 0

    def sum(self):
        assert np.sum(self.tree[-self.capacity:])-self.tree[0] < 0.1, 'sum is {} but root is {}'.format(np.sum(self.tree[-self.capacity:]), self.tree[0])
        return self.tree[0]

    def __getitem__(self, idx: int):
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity-1+idx]

    def batch_sample(self, batch_size: int):
        p_sum = self.tree[0]
        interval = p_sum/batch_size
        prefixsums = np.arange(0, p_sum, interval, dtype=np.float64) + np.random.uniform(0, interval, batch_size)
        indices = np.zeros(batch_size, dtype=int)
        for _ in range(self.layer-1):
            nodes = self.tree[indices*2+1]
            indices = np.where(prefixsums<nodes, indices*2+1, indices*2+2)
            prefixsums = np.where(indices%2==0, prefixsums-self.tree[indices-1], prefixsums)
        
        priorities = self.tree[indices]
        indices -= self.capacity-1

        assert np.all(priorities>0), 'idx: {}, priority: {}'.format(indices, priorities)
        assert np.all(indices>=0) and np.all(indices<self.capacity)

        return indices, priorities

    def batch_update(self, indices: np.ndarray, priorities: np.ndarray):
        indices += self.capacity-1
        self.tree[indices] = priorities

        for _ in range(self.layer-1):
            indices = (indices-1) // 2
            indices = np.unique(indices)
            self.tree[indices] = self.tree[2*indices+1] + self.tree[2*indices+2]
        
        # check
        assert np.sum(self.tree[-self.capacity:])-self.tree[0] < 0.1, 'sum is {} but root is {}'.format(np.sum(self.tree[-self.capacity:]), self.tree[0])


class LocalBuffer:
    def __init__(self, worker_id: int, num_agents: int, map_len: int, init_obs: np.ndarray):
        """
        buffer for each episode
        """
        self.worker_id = worker_id
        self.num_agents = num_agents
        self.map_len = map_len
        self.capacity = config['max_episode_length']
        obs_radius = config['obs_radius']
        self.obs_shape = (6, 2 * obs_radius + 1, 2 * obs_radius + 1)
        self.action_dim = config['action_dim']
        self.hidden_dim = config['hidden_dim']
        self.forward_steps = config['forward_steps']

        self.obs_buf = np.zeros((self.capacity+1, num_agents, *self.obs_shape), dtype=np.bool)
        self.act_buf = np.zeros((self.capacity), dtype=np.uint8)
        self.rew_buf = np.zeros((self.capacity), dtype=np.float16)
        self.hid_buf = np.zeros((self.capacity, num_agents, self.hidden_dim), dtype=np.float16)
        self.comm_mask_buf = np.zeros((self.capacity+1, num_agents, num_agents), dtype=np.bool)
        self.q_buf = np.zeros((self.capacity+1, self.action_dim), dtype=np.float32)

        self.size = 0
        self.obs_buf[0] = init_obs
    
    def __len__(self):
        return self.size

    def add(self, q_val: np.ndarray, action: int, reward: float, next_obs: np.ndarray, hidden: np.ndarray, comm_mask: np.ndarray):
        assert self.size < self.capacity
        self.act_buf[self.size] = action
        self.rew_buf[self.size] = reward
        self.obs_buf[self.size+1] = next_obs
        self.q_buf[self.size] = q_val
        self.hid_buf[self.size] = hidden
        self.comm_mask_buf[self.size] = comm_mask

        self.size += 1

    def finish(self, last_q_val=None, last_comm_mask=None):
        # last q value is None if done
        if last_q_val is None:
            done = True
        else:
            done = False
            self.q_buf[self.size] = last_q_val
            self.comm_mask_buf[self.size] = last_comm_mask
        
        self.obs_buf = self.obs_buf[:self.size+1]
        self.act_buf = self.act_buf[:self.size]
        self.rew_buf = self.rew_buf[:self.size]
        self.hid_buf = self.hid_buf[:self.size]
        self.q_buf = self.q_buf[:self.size+1]
        self.comm_mask_buf = self.comm_mask_buf[:self.size+1]

        # caculate td errors for prioritized experience replay
        td_errors = np.zeros(self.capacity, dtype=np.float32)
        q_max = np.max(self.q_buf[:self.size], axis=1)
        ret = self.rew_buf.tolist() + [ 0 for _ in range(self.forward_steps-1)]
        reward = np.convolve(ret, [0.99**(self.forward_steps-1-i) for i in range(self.forward_steps)],'valid')+q_max
        q_val = self.q_buf[np.arange(self.size), self.act_buf]
        td_errors[:self.size] = np.abs(reward-q_val).clip(1e-4)

        return  self.worker_id, self.num_agents, self.map_len, self.obs_buf, self.act_buf, self.rew_buf, self.hid_buf, td_errors, done, self.size, self.comm_mask_buf


@ray.remote(num_cpus=1)
class GlobalBuffer:
    def __init__(self):
        self.global_capacity = config['global_capacity']
        self.local_capacity = config['max_episode_length']
        self.default_env_setting = tuple(config['default_env_setting'])
        self.action_dim = config['action_dim']
        self.hidden_dim = config['hidden_dim']
        self.max_comm_agents = config['max_comm_agents']
        self.alpha = config['alpha_pr']
        self.beta = config['beta_pr']
        obs_radius = config['obs_radius']
        self.obs_shape = (6, 2 * obs_radius + 1, 2 * obs_radius + 1)
        self.batch_size = config['batch_size']
        self.seq_len = config['seq_len']
        self.forward_steps = config['forward_steps']
        self.max_num_agents = config['max_num_agents']
        self.max_map_length = config['max_map_length']
        self.upgrade_rate = config['upgrade_rate']
        self.learning_threshold = config['learning_threshold']
        self.size = 0
        self.ptr = 0

        # prioritized experience replay
        self.priority_tree = SumTree(self.global_capacity*self.local_capacity)

        self.counter = 0
        self.batched_data = []
        self.stat_dict = {self.default_env_setting:[]}
        self.lock = threading.Lock()
        self.env_setting_set = ray.put([self.default_env_setting])

        self.obs_buf = np.zeros(((self.local_capacity+1)*self.global_capacity, self.max_num_agents, *self.obs_shape), dtype=np.bool)
        self.act_buf = np.zeros((self.local_capacity*self.global_capacity), dtype=np.uint8)
        self.rew_buf = np.zeros((self.local_capacity*self.global_capacity), dtype=np.float16)
        self.hid_buf = np.zeros((self.local_capacity*self.global_capacity, self.max_num_agents, self.hidden_dim), dtype=np.float16)
        self.done_buf = np.zeros(self.global_capacity, dtype=np.bool)
        self.size_buf = np.zeros(self.global_capacity, dtype=np.uint)
        self.comm_mask_buf = np.zeros(((self.local_capacity+1)*self.global_capacity, self.max_num_agents, self.max_num_agents), dtype=np.bool)


    def __len__(self):
        return self.size

    def run(self):
        self.background_thread = threading.Thread(target=self.prepare_data, daemon=True)
        self.background_thread.start()

    def prepare_data(self):
        while True:
            if len(self.batched_data) <= 4:
                data = self.sample_batch(self.batch_size)
                data_id = ray.put(data)
                self.batched_data.append(data_id)
            else:
                time.sleep(0.1)
        
    def get_data(self):
        if len(self.batched_data) == 0:
            print('no prepared data')
            data = self.sample_batch(self.batch_size)
            data_id = ray.put(data)
            return data_id
        else:
            return self.batched_data.pop(0)

    def add(self, data: Tuple):
        '''
        data: worker_id 0, num_agents 1, map_len 2, obs_buf 3, act_buf 4, rew_buf 5, hid_buf 6, td_errors 7, done 8, size 9, comm_mask 10
        '''
        if data[0] >= 12:
            stat_key = (data[1], data[2])
            if stat_key in self.stat_dict:
                self.stat_dict[stat_key].append(data[8])
                if len(self.stat_dict[stat_key]) == 201:
                    self.stat_dict[stat_key].pop(0)

        with self.lock:
            indices = np.arange(self.ptr*self.local_capacity, (self.ptr+1)*self.local_capacity)
            start_idx = self.ptr*self.local_capacity
            # update buffer size
            self.size -= self.size_buf[self.ptr].item()
            self.size += data[9]
            self.counter += data[9]
            self.priority_tree.batch_update(indices, data[7]**self.alpha)

            self.obs_buf[start_idx+self.ptr:start_idx+self.ptr+data[9]+1, :data[1]] = data[3]
            self.act_buf[start_idx:start_idx+data[9]] = data[4]
            self.rew_buf[start_idx:start_idx+data[9]] = data[5]
            self.hid_buf[start_idx:start_idx+data[9], :data[1]] = data[6]
            self.done_buf[self.ptr] = data[8]
            self.size_buf[self.ptr] = data[9]
            self.comm_mask_buf[start_idx+self.ptr:start_idx+self.ptr+data[9]+1] = 0
            self.comm_mask_buf[start_idx+self.ptr:start_idx+self.ptr+data[9]+1, :data[1], :data[1]] = data[10]
            self.ptr = (self.ptr+1) % self.global_capacity

    def sample_batch(self, batch_size: int) -> Tuple:
        b_obs, b_action, b_reward, b_done, b_steps, b_seq_len, b_comm_mask = [], [], [], [], [], [], []
        indices, priorities = [], []
        b_hidden = []
        with self.lock:
            indices, priorities = self.priority_tree.batch_sample(batch_size)
            global_indices = indices // self.local_capacity
            local_indices = indices % self.local_capacity
            for idx, global_idx, local_idx in zip(indices.tolist(), global_indices.tolist(), local_indices.tolist()):
                assert local_idx < self.size_buf[global_idx], 'index is {} but size is {}'.format(local_idx, self.size_buf[global_idx])
                steps = min(self.forward_steps, (self.size_buf[global_idx].item()-local_idx))
                seq_len = min(local_idx+1, self.seq_len)
                if local_idx < self.seq_len-1:
                    obs = self.obs_buf[global_idx*(self.local_capacity+1):idx+global_idx+1+steps]
                    comm_mask = self.comm_mask_buf[global_idx*(self.local_capacity+1):idx+global_idx+1+steps]
                    hidden = np.zeros((self.max_num_agents, self.hidden_dim), dtype=np.float16)
                elif local_idx == self.seq_len-1:
                    obs = self.obs_buf[idx+global_idx+1-self.seq_len:idx+global_idx+1+steps]
                    comm_mask = self.comm_mask_buf[global_idx*(self.local_capacity+1):idx+global_idx+1+steps]
                    hidden = np.zeros((self.max_num_agents, self.hidden_dim), dtype=np.float16)
                else:
                    obs = self.obs_buf[idx+global_idx+1-self.seq_len:idx+global_idx+1+steps]
                    comm_mask = self.comm_mask_buf[idx+global_idx+1-self.seq_len:idx+global_idx+1+steps]
                    hidden = self.hid_buf[idx-self.seq_len]

                if obs.shape[0] < self.seq_len + self.forward_steps:
                    pad_len = self.seq_len + self.forward_steps-obs.shape[0]
                    obs = np.pad(obs, ((0,pad_len),(0,0),(0,0),(0,0),(0,0)))
                    comm_mask = np.pad(comm_mask, ((0,pad_len),(0,0),(0,0)))

                action = self.act_buf[idx]
                reward = 0
                for i in range(steps):
                    reward += self.rew_buf[idx+i]*0.99**i
                if self.done_buf[global_idx] and local_idx >= self.size_buf[global_idx]-self.forward_steps:
                    done = True
                else:
                    done = False
                
                b_obs.append(obs)
                b_action.append(action)
                b_reward.append(reward)
                b_done.append(done)
                b_steps.append(steps)
                b_seq_len.append(seq_len)
                b_hidden.append(hidden)
                b_comm_mask.append(comm_mask)

            # importance sampling weight
            min_p = np.min(priorities)
            weights = np.power(priorities/min_p, -self.beta)

            data = (
                torch.from_numpy(np.stack(b_obs).astype(np.float16)),
                torch.LongTensor(b_action).unsqueeze(1),
                torch.HalfTensor(b_reward).unsqueeze(1),
                torch.HalfTensor(b_done).unsqueeze(1),
                torch.HalfTensor(b_steps).unsqueeze(1),
                torch.LongTensor(b_seq_len),
                torch.from_numpy(np.concatenate(b_hidden)),
                torch.from_numpy(np.stack(b_comm_mask)),
                indices,
                torch.from_numpy(weights).unsqueeze(1),
                self.ptr
            )
            return data

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray, old_ptr: int):
        """Update priorities of sampled transitions"""
        with self.lock:

            # discard the indices that already been discarded in replay buffer during training
            if self.ptr > old_ptr:
                # range from [old_ptr, self.ptr)
                mask = (indices < old_ptr*self.local_capacity) | (indices >= self.ptr*self.local_capacity)
                indices = indices[mask]
                priorities = priorities[mask]
            elif self.ptr < old_ptr:
                # range from [0, self.ptr) & [old_ptr, self,capacity)
                mask = (indices < old_ptr*self.local_capacity) & (indices >= self.ptr*self.local_capacity)
                indices = indices[mask]
                priorities = priorities[mask]
            self.priority_tree.batch_update(np.copy(indices), np.copy(priorities)**self.alpha)

    def stats(self, interval: int):
        print('buffer update speed: {}/s'.format(self.counter/interval))
        print('buffer size: {}'.format(self.size))

        print('  ', end='')
        for i in range(self.default_env_setting[1], self.max_map_length+1, 5):
            print('   {:2d}   '.format(i), end='')
        print()

        for num_agents in range(self.default_env_setting[0], self.max_num_agents+1):
            print('{:2d}'.format(num_agents), end='')
            for map_len in range(self.default_env_setting[1], self.max_map_length+1, 5):
                if (num_agents, map_len) in self.stat_dict:
                    print('{:4d}/{:<3d}'.format(sum(self.stat_dict[(num_agents, map_len)]), len(self.stat_dict[(num_agents, map_len)])), end='')
                else:
                    print('   N/A  ', end='')
            print()

        for key, val in self.stat_dict.copy().items():
            # print('{}: {}/{}'.format(key, sum(val), len(val)))
            if len(val) == 200 and sum(val) >= 200 * self.upgrade_rate:
                # add number of agents
                add_agent_key = (key[0]+1, key[1]) 
                if add_agent_key[0] <= self.max_num_agents and add_agent_key not in self.stat_dict:
                    self.stat_dict[add_agent_key] = []
                
                if key[1] < self.max_map_lenght:
                    add_map_key = (key[0], key[1]+5) 
                    if add_map_key not in self.stat_dict:
                        self.stat_dict[add_map_key] = []
                
        self.env_setting_set = ray.put(list(self.stat_dict.keys()))
        self.counter = 0

    def ready(self):
        if len(self) >= self.learning_threshold:
            return True
        else:
            return False
    
    def get_env_setting_set(self):
        return self.env_setting_set

    def check_done(self):
        for i in range(self.max_num_agents):
            if (i+1, self.max_map_length) not in self.stat_dict:
                return False
            l = self.stat_dict[(i+1, self.max_map_length)]
            if len(l) < 200:
                return False
            elif sum(l) < 200*self.upgrade_rate:
                return False  
        return True