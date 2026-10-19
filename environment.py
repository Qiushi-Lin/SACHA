import random
from collections import defaultdict
from typing import List
import numpy as np

from utils import load_config, map_partition, sample_starts_goals, bfs_distance_maps, build_observations, FREE_SPACE, OBSTACLE


def setting_feasible(num_agents, map_length):
    return 4 * num_agents <= map_length * map_length


class POMAPFEnv:
    def __init__(self, config=None):
        self.config = config if config is not None else load_config()
        c = self.config
        self.action_mapping = np.array([c['action_mapping'][i] for i in range(c['action_dim'])], dtype=int)
        self.obs_radius = c['obs_radius']
        self.K = c['K_obs']
        self.heuristic_norm = c['heuristic_norm']
        self.reward_fn = c['reward_fn']
        self.reward_shaping = c['reward_shaping']
        self.gamma = c['gamma']
        self.lambda_hurl = c['lambda_hurl']
        self.obstacle_density = c['obstacle_density']
        self.env_setting_set = [tuple(c['default_env_setting'])]
        self.num_agents = 0
        self.num_steps = 0

    def update_env_setting_set(self, new_env_setting_set):
        self.env_setting_set = [tuple(setting) for setting in new_env_setting_set]

    def reset(self, setting=None):
        num_agents, map_length = setting if setting is not None else random.choice(self.env_setting_set)
        grid_map, starts, goals = self.generate_instance(num_agents, map_length)
        self.load(grid_map, starts, goals)
        return self.observe()

    def generate_instance(self, num_agents, map_length, max_attempts=100):
        if not setting_feasible(num_agents, map_length):
            raise ValueError('{} agents do not fit in a {}x{} map'.format(num_agents, map_length, map_length))
        for attempt in range(max_attempts):
            density = np.random.triangular(*self.obstacle_density) * (1 - attempt / max_attempts)
            grid_map = np.where(np.random.rand(map_length, map_length) < density, OBSTACLE, FREE_SPACE)
            partitions = [p for p in map_partition(grid_map) if len(p) >= 2]
            if sum(len(p) // 2 for p in partitions) < num_agents:
                continue
            starts, goals = sample_starts_goals(partitions, num_agents)
            return grid_map, starts, goals
        raise RuntimeError('failed to generate an instance with {} agents on a {}x{} map'.format(num_agents, map_length, map_length))

    def load(self, grid_map: np.ndarray, agents_pos: np.ndarray, goals_pos: np.ndarray):
        self.map = np.array(grid_map, dtype=int)
        self.map_size = (self.map.shape[0], self.map.shape[1])
        self.agents_pos = np.array(agents_pos, dtype=int).reshape(-1, 2)
        self.goals_pos = np.array(goals_pos, dtype=int).reshape(-1, 2)
        self.num_agents = self.agents_pos.shape[0]
        self.num_steps = 0
        self.get_heuristic_map()

    def get_heuristic_map(self):
        r = self.obs_radius
        self.dist = bfs_distance_maps(self.map == FREE_SPACE, self.goals_pos)
        finite = np.where(np.isfinite(self.dist), self.dist, 0)
        self.dmax = np.maximum(finite.reshape(self.num_agents, -1).max(1), 1).astype(np.float32)
        self.padded_map = np.pad(self.map != FREE_SPACE, r, 'constant', constant_values=True)
        self.padded_dist = np.pad(self.dist, ((0, 0), (r, r), (r, r)), 'constant', constant_values=np.inf)

    def heuristic(self):
        d = self.dist[np.arange(self.num_agents), self.agents_pos[:, 0], self.agents_pos[:, 1]]
        return -d / self.dmax

    def step(self, actions: List[int]):
        actions = np.asarray(actions, dtype=int).reshape(-1)
        assert len(actions) == self.num_agents, 'only {} actions as input while {} agents in environment'.format(len(actions), self.num_agents)
        assert np.all((actions >= 0) & (actions < len(self.action_mapping))), 'action index out of range'
        height, width = self.map_size

        at_goal = np.all(self.agents_pos == self.goals_pos, axis=1)
        moving = actions != 0
        rewards = np.where(moving, self.reward_fn['move'],
                           np.where(at_goal, self.reward_fn['stay_on_goal'], self.reward_fn['stay_off_goal'])).astype(np.float32)
        next_pos = self.agents_pos + self.action_mapping[actions]

        def revert(ids):
            next_pos[ids] = self.agents_pos[ids]
            rewards[ids] = self.reward_fn['collision']
            checking[ids] = False

        checking = moving.copy()
        out_of_map = (next_pos[:, 0] < 0) | (next_pos[:, 0] >= height) | (next_pos[:, 1] < 0) | (next_pos[:, 1] >= width)
        blocked = out_of_map.copy()
        inside = ~out_of_map
        blocked[inside] = self.map[next_pos[inside, 0], next_pos[inside, 1]] == OBSTACLE
        revert(np.flatnonzero(blocked & moving))

        pos_to_agent = {tuple(p): i for i, p in enumerate(self.agents_pos.tolist())}
        for i in np.flatnonzero(checking):
            if not checking[i]:
                continue
            j = pos_to_agent.get(tuple(next_pos[i]))
            if j is not None and j != i and checking[j] and np.array_equal(next_pos[j], self.agents_pos[i]):
                revert([i, j])

        conflict = True
        while conflict:
            conflict = False
            occupants = defaultdict(list)
            for i, p in enumerate(next_pos.tolist()):
                occupants[tuple(p)].append(i)
            for ids in occupants.values():
                if len(ids) < 2:
                    continue
                movers = [i for i in ids if checking[i]]
                losers = movers[1:] if len(movers) == len(ids) else movers
                if losers:
                    revert(losers)
                    conflict = True

        self.agents_pos = next_pos
        self.num_steps += 1

        if np.unique(self.agents_pos, axis=0).shape[0] < self.num_agents:
            raise RuntimeError('agents overlap after step {}'.format(self.num_steps))

        done = bool(np.array_equal(self.agents_pos, self.goals_pos))
        if done:
            rewards[:] = self.reward_fn['reach_goal']
        raw_rewards = rewards.copy()

        if self.reward_shaping:
            rewards = rewards + (1 - self.lambda_hurl) * self.gamma * self.heuristic().astype(np.float32)

        info = {'step': self.num_steps - 1, 'raw_rewards': raw_rewards, 'success': done}
        return self.observe(), rewards, done, info

    def observe(self):
        return build_observations(self.padded_map, self.padded_dist, self.dmax, self.agents_pos,
                                  self.K, self.obs_radius, self.heuristic_norm)
