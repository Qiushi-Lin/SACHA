import random
from typing import List
import numpy as np

from utils import map_partition

# config
import yaml
config = yaml.safe_load(open("./config.yaml", 'r'))


class POMAPFEnv:
    def __init__(self):
        self.action_mapping = config['action_mapping']
        self.default_env_setting = config['default_env_setting']
        self.env_setting_set = [self.default_env_setting]
        self.num_agents = self.default_env_setting[0]
        self.map_size = (self.default_env_setting[1], self.default_env_setting[1])

        self.obstacle_density = np.random.triangular(0, 0.33, 0.5)
        self.map = np.random.choice(2, self.map_size, p=[1-self.obstacle_density, self.obstacle_density]).astype(int)
        partition_list = map_partition(self.map)
        partition_list = [ partition for partition in partition_list if len(partition) >= 2 ]
        while len(partition_list) == 0:
            self.map = np.random.choice(2, self.map_size, p=[1-self.obstacle_density, self.obstacle_density]).astype(int)
            partition_list = map_partition(self.map)
            partition_list = [ partition for partition in partition_list if len(partition) >= 2 ]
        
        self.agents_pos = np.empty((self.num_agents, 2), dtype=int)
        self.goals_pos = np.empty((self.num_agents, 2), dtype=int)
        pos_num = sum([ len(partition) for partition in partition_list ])
        
        # loop to assign agent original position and goal position for each agent
        for i in range(self.num_agents):
            pos_idx = random.randint(0, pos_num-1)
            partition_idx = 0
            for partition in partition_list:
                if pos_idx >= len(partition):
                    pos_idx -= len(partition)
                    partition_idx += 1
                else:
                    break 
            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.agents_pos[i] = np.asarray(pos, dtype=int)
            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.goals_pos[i] = np.asarray(pos, dtype=int)
            partition_list = [partition for partition in partition_list if len(partition) >= 2]
            pos_num = sum([ len(partition) for partition in partition_list ])

        self.obs_radius = config['obs_radius']
        self.reward_fn = config['reward_fn']
        self.get_heuristic_map()
        self.num_steps = 0
        self.last_actions = np.zeros((self.num_agents, 5, 2*self.obs_radius+1, 2*self.obs_radius+1), dtype=np.bool)

    def update_env_setting_set(self, new_env_setting_set):
        self.env_setting_set = new_env_setting_set

    def reset(self):
        rand_env_setting = random.choice(self.env_setting_set)
        self.num_agents = rand_env_setting[0]
        self.map_size = (rand_env_setting[1], rand_env_setting[1])
        self.obstacle_density = np.random.triangular(0, 0.33, 0.5)
        
        self.map = np.random.choice(2, self.map_size, p=[1-self.obstacle_density, self.obstacle_density]).astype(np.float32)
        partition_list = map_partition(self.map)
        partition_list = [ partition for partition in partition_list if len(partition) >= 2 ]
        while len(partition_list) == 0:
            self.map = np.random.choice(2, self.map_size, p=[1-self.obstacle_density, self.obstacle_density]).astype(np.float32)
            partition_list = map_partition(self.map)
            partition_list = [ partition for partition in partition_list if len(partition) >= 2 ]
        self.agents_pos = np.empty((self.num_agents, 2), dtype=int)
        self.goals_pos = np.empty((self.num_agents, 2), dtype=int)
        pos_num = sum([ len(partition) for partition in partition_list ])
        
        for i in range(self.num_agents):
            pos_idx = random.randint(0, pos_num-1)
            partition_idx = 0
            for partition in partition_list:
                if pos_idx >= len(partition):
                    pos_idx -= len(partition)
                    partition_idx += 1
                else:
                    break 
            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.agents_pos[i] = np.asarray(pos, dtype=int)
            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.goals_pos[i] = np.asarray(pos, dtype=int)
            partition_list = [ partition for partition in partition_list if len(partition) >= 2 ]
            pos_num = sum([ len(partition) for partition in partition_list ])
        self.num_steps = 0
        self.get_heuristic_map()
        self.last_actions = np.zeros((self.num_agents, 5, 2*self.obs_radius+1, 2*self.obs_radius+1), dtype=np.bool)

        return self.observe()

    def load(self, map:np.ndarray, agents_pos:np.ndarray, goals_pos:np.ndarray):
        self.map = np.copy(map)
        self.agents_pos = np.copy(agents_pos)
        self.goals_pos = np.copy(goals_pos)
        self.num_agents = agents_pos.shape[0]
        self.map_size = (self.map.shape[0], self.map.shape[1])
        self.num_steps = 0
        self.get_heuristic_map()
        self.last_actions = np.zeros((self.num_agents, 5, 2*self.obs_radius+1, 2*self.obs_radius+1), dtype=np.bool)

    def get_heuristic_map(self):
        dist_map = np.ones((self.num_agents, *self.map_size), dtype=int) * float('inf')
        for i in range(self.num_agents):
            open_list = list()
            x, y = tuple(self.goals_pos[i])
            open_list.append((x, y))
            dist_map[i, x, y] = 0

            while open_list:
                x, y = open_list.pop(0)
                dist = dist_map[i, x, y]
                up = x-1, y
                if up[0] >= 0 and self.map[up]==0 and dist_map[i, x-1, y] > dist+1:
                    dist_map[i, x-1, y] = dist+1
                    if up not in open_list:
                        open_list.append(up)
                down = x+1, y
                if down[0] < self.map_size[0] and self.map[down]==0 and dist_map[i, x+1, y] > dist+1:
                    dist_map[i, x+1, y] = dist+1
                    if down not in open_list:
                        open_list.append(down)
                left = x, y-1
                if left[1] >= 0 and self.map[left]==0 and dist_map[i, x, y-1] > dist+1:
                    dist_map[i, x, y-1] = dist+1
                    if left not in open_list:
                        open_list.append(left)
                right = x, y+1
                if right[1] < self.map_size[1] and self.map[right]==0 and dist_map[i, x, y+1] > dist+1:
                    dist_map[i, x, y+1] = dist+1
                    if right not in open_list:
                        open_list.append(right)
        self.heuri_map = np.zeros((self.num_agents, 4, *self.map_size), dtype=np.bool)

        for x in range(self.map_size[0]):
            for y in range(self.map_size[1]):
                if self.map[x, y] == 0:
                    for i in range(self.num_agents):
                        if x > 0 and dist_map[i, x-1, y] < dist_map[i, x, y]:
                            assert dist_map[i, x-1, y] == dist_map[i, x, y]-1
                            self.heuri_map[i, 0, x, y] = 1
                        if x < self.map_size[0]-1 and dist_map[i, x+1, y] < dist_map[i, x, y]:
                            assert dist_map[i, x+1, y] == dist_map[i, x, y]-1
                            self.heuri_map[i, 1, x, y] = 1
                        if y > 0 and dist_map[i, x, y-1] < dist_map[i, x, y]:
                            assert dist_map[i, x, y-1] == dist_map[i, x, y]-1
                            self.heuri_map[i, 2, x, y] = 1
                        if y < self.map_size[1]-1 and dist_map[i, x, y+1] < dist_map[i, x, y]:
                            assert dist_map[i, x, y+1] == dist_map[i, x, y]-1
                            self.heuri_map[i, 3, x, y] = 1
        self.heuri_map = np.pad(self.heuri_map, ((0, 0), (0, 0), (self.obs_radius, self.obs_radius), (self.obs_radius, self.obs_radius)))

    def step(self, actions: List[int]):
        assert len(actions) == self.num_agents, 'only {} actions as input while {} agents in environment'.format(len(actions), self.num_agents)
        assert all([action_idx<5 and action_idx>=0 for action_idx in actions]), 'action index out of range'

        checking_list = [i for i in range(self.num_agents)]

        rewards = []
        next_pos = np.copy(self.agents_pos)

        # remove unmoving agent id
        for agent_id in checking_list.copy():
            if actions[agent_id] == 0:
                # unmoving
                if np.array_equal(self.agents_pos[agent_id], self.goals_pos[agent_id]):
                    rewards.append(self.reward_fn['stay_on_goal'])
                else:
                    rewards.append(self.reward_fn['stay_off_goal'])
                checking_list.remove(agent_id)
            else:
                # move
                next_pos[agent_id] += self.action_mapping[actions[agent_id]]
                rewards.append(self.reward_fn['move'])

        # first round check, these two conflicts have the heightest priority
        for agent_id in checking_list.copy():
            if np.any(next_pos[agent_id]<0) or np.any(next_pos[agent_id]>=self.map_size[0]):
                # agent out of map range
                rewards[agent_id] = self.reward_fn['collision']
                next_pos[agent_id] = self.agents_pos[agent_id]
                checking_list.remove(agent_id)
            elif self.map[tuple(next_pos[agent_id])] == 1:
                # collide obstacle
                rewards[agent_id] = self.reward_fn['collision']
                next_pos[agent_id] = self.agents_pos[agent_id]
                checking_list.remove(agent_id)

        # second round check, agent swapping conflict
        no_conflict = False
        while not no_conflict:
            no_conflict = True
            for agent_id in checking_list:
                target_agent_id = np.where(np.all(next_pos[agent_id]==self.agents_pos, axis=1))[0]
                if target_agent_id:
                    target_agent_id = target_agent_id.item()
                    assert target_agent_id != agent_id, 'logic bug'
                    if np.array_equal(next_pos[target_agent_id], self.agents_pos[agent_id]):
                        assert target_agent_id in checking_list, 'target_agent_id should be in checking list'
                        next_pos[agent_id] = self.agents_pos[agent_id]
                        rewards[agent_id] = self.reward_fn['collision']
                        next_pos[target_agent_id] = self.agents_pos[target_agent_id]
                        rewards[target_agent_id] = self.reward_fn['collision']
                        checking_list.remove(agent_id)
                        checking_list.remove(target_agent_id)
                        no_conflict = False
                        break

        # third round check, agent collision conflict
        no_conflict = False
        while not no_conflict:
            no_conflict = True
            for agent_id in checking_list:
                collide_agent_id = np.where(np.all(next_pos==next_pos[agent_id], axis=1))[0].tolist()
                if len(collide_agent_id) > 1:
                    # collide agent
                    # if all agents in collide agent are in checking list
                    all_in_checking = True
                    for id in collide_agent_id.copy():
                        if id not in checking_list:
                            all_in_checking = False
                            collide_agent_id.remove(id)
                    if all_in_checking:
                        collide_agent_pos = next_pos[collide_agent_id].tolist()
                        for pos, id in zip(collide_agent_pos, collide_agent_id):
                            pos.append(id)
                        collide_agent_pos.sort(key=lambda x: x[0]*self.map_size[0]+x[1])
                        collide_agent_id.remove(collide_agent_pos[0][2])
                    next_pos[collide_agent_id] = self.agents_pos[collide_agent_id]
                    for id in collide_agent_id:
                        rewards[id] = self.reward_fn['collision']
                    for id in collide_agent_id:
                        checking_list.remove(id)
                    no_conflict = False
                    break

        self.agents_pos = np.copy(next_pos)
        self.num_steps += 1

        # check done
        if np.array_equal(self.agents_pos, self.goals_pos):
            done = True
            rewards = [self.reward_fn['reach_goal'] for _ in range(self.num_agents)]
        else:
            done = False
        info = {'step': self.num_steps-1}

        # make sure no overlapping agents
        if np.unique(self.agents_pos, axis=0).shape[0] < self.num_agents:
            print(self.num_steps)
            print(self.map)
            print(self.agents_pos)
            raise RuntimeError('unique')

        # update last actions
        self.last_actions = np.zeros((self.num_agents, 5, 2*self.obs_radius+1, 2*self.obs_radius+1), dtype=np.bool)
        self.last_actions[np.arange(self.num_agents), np.array(actions)] = 1

        return self.observe(), rewards, done, info


    def observe(self):
        obs = np.zeros((self.num_agents, 6, 2*self.obs_radius+1, 2*self.obs_radius+1), dtype=np.bool)
        obstacle_map = np.pad(self.map, self.obs_radius, 'constant', constant_values=0)
        agent_map = np.zeros((self.map_size), dtype=np.bool)
        agent_map[self.agents_pos[:,0], self.agents_pos[:,1]] = 1
        agent_map = np.pad(agent_map, self.obs_radius, 'constant', constant_values=0)
        for i, agent_pos in enumerate(self.agents_pos):
            x, y = agent_pos
            obs[i, 0] = agent_map[x:x+2*self.obs_radius+1, y:y+2*self.obs_radius+1]
            obs[i, 0, self.obs_radius, self.obs_radius] = 0
            obs[i, 1] = obstacle_map[x:x+2*self.obs_radius+1, y:y+2*self.obs_radius+1]
            obs[i, 2:] = self.heuri_map[i, :, x:x+2*self.obs_radius+1, y:y+2*self.obs_radius+1]

        return obs, np.copy(self.agents_pos)