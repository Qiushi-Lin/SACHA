import random
from copy import deepcopy
import ray
import torch
import numpy as np
from model import PolicyNet
from environment import POMAPFEnv
from buffer import LocalBuffer, GlobalBuffer
from learner import Learner

# config
import yaml
config = yaml.safe_load(open("./config.yaml", 'r'))


@ray.remote(num_cpus=1)
class Worker:
    def __init__(self, worker_id: int, epsilon: float, learner: Learner, buffer: GlobalBuffer):
        self.id = worker_id
        self.model = PolicyNet()
        self.model.eval()
        self.env = POMAPFEnv()
        self.epsilon = epsilon
        self.learner = learner
        self.global_buffer = buffer
        self.max_episode_length = config['max_episode_length']
        self.steps_per_update = config['steps_per_update']
        self.counter = 0

    def run(self):
        done = False
        obs, pos, local_buffer = self.reset()
        
        while True:
            # sample action
            actions, q_val, hidden, comm_mask = self.model.step(torch.from_numpy(obs.astype(np.float32)), torch.from_numpy(pos.astype(np.float32)))

            if random.random() < self.epsilon:
                # Note: only one agent do random action in order to keep the environment stable
                actions[0] = np.random.randint(0, 5)
            # take action in env
            (next_obs, next_pos), rewards, done, _ = self.env.step(actions)
            # return data and update observation
            local_buffer.add(q_val[0], actions[0], rewards[0], next_obs, hidden, comm_mask)

            if done == False and self.env.num_steps < self.max_episode_length:
                obs, pos = next_obs, next_pos
            else:
                # finish and send buffer
                if done:
                    data = local_buffer.finish()
                else:
                    _, q_val, hidden, comm_mask = self.model.step(torch.from_numpy(next_obs.astype(np.float32)), torch.from_numpy(next_pos.astype(np.float32)))
                    data = local_buffer.finish(q_val[0], comm_mask)

                self.global_buffer.add.remote(data)
                done = False
                obs, pos, local_buffer = self.reset()

            self.counter += 1
            if self.counter == config['steps_per_update']:
                self.update_weights()
                self.counter = 0

    def update_weights(self):
        '''load weights from learner'''
        # update network parameters
        weights_id = ray.get(self.learner.get_weights.remote())
        weights = ray.get(weights_id)
        self.model.load_state_dict(weights)
        # update environment settings set (number of agents and map size)
        new_env_settings_set = ray.get(self.global_buffer.get_env_setting_set.remote())
        self.env.update_env_setting_set(ray.get(new_env_settings_set))
    
    def reset(self):
        self.model.reset()
        obs, pos = self.env.reset()
        num_agents, map_len = self.env.num_agents, self.env.map_size[0]
        local_buffer = LocalBuffer(self.id, num_agents, map_len, obs)
        return obs, pos, local_buffer