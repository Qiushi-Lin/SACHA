import random
import numpy as np
import torch

from model import SACHAActor
from environment import POMAPFEnv
from buffer import compress_dist


class Worker:
    def __init__(self, worker_id: int, config, learner=None, buffer=None, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.id = worker_id
        self.config = config
        self.actor = SACHAActor(config)
        self.actor.eval()
        self.env = POMAPFEnv(config)
        self.learner = learner
        self.global_buffer = buffer
        self.max_episode_length = config['max_episode_length']
        self.steps_per_sync = config['steps_per_sync']

    def set_weights(self, state_dict):
        self.actor.load_state_dict(state_dict)

    def set_env_setting_set(self, env_setting_set):
        self.env.update_env_setting_set(env_setting_set)

    def run_episode(self, greedy=False):
        obs, nbr, adj = self.env.reset()
        self.actor.reset()
        positions = [self.env.agents_pos.copy()]
        actions, rewards = [], []
        done = False
        while not done and self.env.num_steps < self.max_episode_length:
            action = self.actor.act(obs, nbr, adj, greedy=greedy)
            (obs, nbr, adj), reward, done, _ = self.env.step(action)
            positions.append(self.env.agents_pos.copy())
            actions.append(action)
            rewards.append(reward)

        return {
            'num_agents': self.env.num_agents,
            'map_len': self.env.map_size[0],
            'padded_map': self.env.padded_map,
            'padded_dist': compress_dist(self.env.padded_dist),
            'dmax': self.env.dmax,
            'pos': np.array(positions, dtype=np.int16),
            'actions': np.array(actions, dtype=np.uint8),
            'rewards': np.array(rewards, dtype=np.float32),
            'done': done,
            'length': len(actions),
        }

    def run(self):
        torch.set_num_threads(1)
        self.sync()
        steps = 0
        while True:
            episode = self.run_episode()
            self.global_buffer.add.remote(episode)
            steps += episode['length']
            if steps >= self.steps_per_sync:
                self.sync()
                steps = 0

    def sync(self):
        import ray
        self.set_weights(ray.get(self.learner.get_weights.remote()))
        self.set_env_setting_set(ray.get(self.global_buffer.get_env_setting_set.remote()))
