import numpy as np
import heapq
import gymnasium as gym
from gym import spaces

from utils.make_instances import generate_random_map, map_partition, generate_random_agents


class POMAPFEnv(gym.Env):
    def __init__(self, config):
        self.OBSTACLE, self.FREE_SPACE = config['grid_map']['OBSTACLE'], config['grid_map']['FREE_SPACE']
        self.fov_size = tuple(config['fov_size'])
        self.obs_r = (int(np.floor(self.fov_size[0]/2)), int(np.floor(self.fov_size[1]/2)))
        self.K_obs = config['K_obs']
        self.ind_reward_func = config['ind_reward_func']
        self.lambda_r = config['lambda_r']
        self.action_mapping = config['action_mapping']
        self.action_dim = config['action_dim']
        self.num_agents = config['default_num_agents']
        self.max_num_agents = config['max_num_agents']
        self.map_size = config['default_map_size']
        self.grid_map = None
        self.curr_state, self.goal_state = None, None
        self.load_random_map()
        self.heuristic_maps = {}
        self._get_heuristic_maps()

    @property
    def observation_space(self):
        return spaces.MultiDiscrete(np.ones((self.max_num_agents, self.K_obs, 3, self.fov_size[0], self.fov_size[1])))

    @property
    def action_space(self):
        return spaces.Discrete(self.max_num_agents, self.action_dim)

    def seed(self, seed=0):
        np.random.seed(seed)

    def reset(self, seed=0):
        self.load_random_map()
        self.seed(seed)
        return self.observe(), None

    def load(self, grid_map, curr_state, goal_state):
        self.grid_map = grid_map
        self.curr_state = curr_state
        self.goal_state = goal_state        
        self.num_agents = len(self.curr_state)
        self.map_size = max(len(self.grid_map), len(self.grid_map[0]))
        self._get_heuristic_maps()

    def get_curr_state(self):
        return self.curr_state

    def get_level(self):
        return self.map_size, self.num_agents

    def set_level(self, map_size, num_agents):
        self.map_size = map_size
        self.num_agents = num_agents
        self.load_random_map()
        self._get_heuristic_maps()

    def load_random_map(self):
        num_obstacles = np.floor(self.map_size * self.map_size * np.random.triangular(0, 0.33, 0.5))
        self.grid_map = generate_random_map(self.map_size, self.map_size, num_obstacles)
        map_partitions = map_partition(self.grid_map)
        self.curr_state, self.goal_state = generate_random_agents(self.grid_map, map_partitions, self.num_agents)

    def _move(self, loc, move):
        action = self.action_mapping[int(move)]
        return (loc[0] + action[0], loc[1] + action[1])

    def _get_heuristic_maps(self):
        self.heuristic_maps = {}
        self.padded_heuristic_maps = {}
        for i in range(self.num_agents):
            self.heuristic_maps[i] = np.zeros(np.shape(self.grid_map))
            goal = self.goal_state[i]
            open_list = []
            closed_list = {}
            root = {'loc': goal, 'cost': 0}
            heapq.heappush(open_list, (root['cost'], goal, root))
            closed_list[goal] = root
            while len(open_list) > 0:
                (cost, loc, curr) = heapq.heappop(open_list)
                for d in range(4):
                    child_loc = self._move(loc, d)
                    child_cost = cost + 1
                    if child_loc[0] < 0 or child_loc[0] >= self.map_size \
                    or child_loc[1] < 0 or child_loc[1] >= self.map_size:
                        continue
                    if self.grid_map[child_loc[0]][child_loc[1]] == self.OBSTACLE:
                        continue
                    child = {'loc': child_loc, 'cost': child_cost}
                    if child_loc in closed_list:
                        existing_node = closed_list[child_loc]
                        if existing_node['cost'] > child_cost:
                            closed_list[child_loc] = child
                            heapq.heappush(open_list, (child_cost, child_loc, child))
                    else:
                        closed_list[child_loc] = child
                        heapq.heappush(open_list, (child_cost, child_loc, child))

            for x in range(self.map_size):
                for y in range(self.map_size):
                    if (x, y) in closed_list:
                        self.heuristic_maps[i][x][y] = closed_list[(x, y)]['cost'] / (self.map_size * self.map_size)
                    else:
                        self.heuristic_maps[i][x][y] = 1.0

    def _detect_vertex_collision(self, path1, path2):
        if np.array_equal(path1[1], path2[1]):
            return True
        return False

    def _detect_edge_collision(self, path1, path2):
        if np.array_equal(path1[1], path2[0]) and np.array_equal(path1[0], path2[1]):
            return True
        return False

    def step(self, action):
        reward = [0 for _ in range(self.num_agents)]
        paths = []
        for i in range(self.num_agents):
            next_state = self.curr_state[i]
            if action[i] == 4:
                if np.array_equal(next_state, self.goal_state[i]):
                    reward[i] = self.ind_reward_func['stay_on_goal']
                else:
                    reward[i] = self.ind_reward_func['stay_off_goal']
            else:
                x, y = self._move(self.curr_state[i], action[i])
                # obstacle check
                if 0 <= x < self.map_size and 0 <= y < self.map_size and self.grid_map[x][y] == self.FREE_SPACE:
                    next_state = (x, y)
                    reward[i] = self.ind_reward_func['move'] - (1 - self.lambda_r) * self.heuristic_maps[i][x][y]
                else:
                    reward[i] = self.ind_reward_func['collision']
            paths.append([self.curr_state[i], next_state])
        # edge collision check
        for i in range(self.num_agents):
            for j in range(i+1, self.num_agents):
                if self._detect_edge_collision(paths[i], paths[j]):
                    paths[i][1] = self.curr_state[i]
                    paths[j][1] = self.curr_state[j]
                    reward[i] = self.ind_reward_func['collision']
                    reward[j] = self.ind_reward_func['collision']
        # vertex collision check
        collision_flag = True
        while collision_flag:
            collision_flag = False
            for i in range(self.num_agents):
                for j in range(i+1, self.num_agents):
                    if self._detect_vertex_collision(paths[i], paths[j]):
                        k = i
                        if np.array_equal(paths[i][0], paths[i][1]):
                            k = j
                        elif np.array_equal(paths[j][0], paths[j][1]):
                            k = i
                        else:
                            x, y = paths[i][1]
                            if self.heuristic_maps[i][x][y] > self.heuristic_maps[j][x][y]:
                                k = j
                        paths[k][1] = self.curr_state[k]
                        reward[k] = self.ind_reward_func['collision']
                        collision_flag = True
                        break

        next_curr_state = []
        for i in range(self.num_agents):
            next_curr_state.append(paths[i][1])
        self.curr_state = next_curr_state
        done = [np.equal(x, y).all() for x, y in zip(self.curr_state, self.goal_state)]
        if all(done):
            reward = [self.ind_reward_func['reach_goal'] for _ in range(self.num_agents)]
        obs = self.observe()
        info = {'num_agents': self.num_agents, 'num_finished_agents': sum(done)}
        return obs, reward, done, None, info

    def observe(self):
        padded_grid_map = np.pad(self.grid_map, ((self.obs_r[0], self.obs_r[0]), (self.obs_r[1], self.obs_r[1])), mode='constant', constant_values=self.OBSTACLE)
        padded_heuristic_maps = {}
        for i in range(self.num_agents):
            padded_heuristic_maps[i] =  np.pad(self.heuristic_maps[i], ((self.obs_r[0], self.obs_r[0]), (self.obs_r[1], self.obs_r[1])), mode='constant', constant_values=1.0)
        observed_agents = {}
        for i in range(self.num_agents):
            p = []
            for j in range(self.num_agents):
                x_i, y_i = self.curr_state[i][0], self.curr_state[i][1]
                x_j, y_j = self.curr_state[j][0], self.curr_state[j][1]
                if abs(x_i - x_j) <= self.obs_r[0] and abs(y_i - y_j) <= self.obs_r[1]:
                    heapq.heappush(p, (abs(x_i - x_j) + abs(y_i - y_j), j))
            observed_agents[i] = []
            for k in range(self.K_obs):
                if len(p):
                    _, a = heapq.heappop(p)
                    observed_agents[i].append(a)
                else:
                    break
        obs = np.zeros((self.num_agents, self.K_obs, 3, self.fov_size[0], self.fov_size[1]))
        for i in range(self.num_agents):
            for k, j in enumerate(observed_agents[i]):
                obs[i][k][0] = padded_grid_map[self.curr_state[i][0]:self.curr_state[i][0]+self.fov_size[0],self.curr_state[i][1]:self.curr_state[i][1]+self.fov_size[1]]
                o_x = self.obs_r[0] - self.curr_state[i][0] + self.curr_state[j][0]
                o_y = self.obs_r[1] - self.curr_state[i][1] + self.curr_state[j][1]
                obs[i][k][1][o_x][o_y] = 1.0
                obs[i][k][2] = padded_heuristic_maps[j][self.curr_state[i][0]:self.curr_state[i][0]+self.fov_size[0],self.curr_state[i][1]:self.curr_state[i][1]+self.fov_size[1]]
        return obs # (num_agents, self.K_obs, 3, fov_size[0], fov_size[1])