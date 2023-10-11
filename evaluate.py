import argparse
import pickle
from tqdm import tqdm
import numpy as np
import torch

from environment import POMAPFEnv
from model import AttentionPolicy

# config
import yaml
config = yaml.safe_load(open("./configs/config_base.yaml", 'r'))
config = config | yaml.safe_load(open("./configs/config_eval.yaml", 'r'))
OBSTACLE, FREE_SPACE = config['grid_map']['OBSTACLE'], config['grid_map']['FREE_SPACE']
num_instances_per_test = config['num_instances_per_test']
test_settings = config['test_settings']
max_timesteps = config['max_timesteps']
hidden_dim = config['hidden_dim']


def test_one_case(model, grid_map, starts, goals, horizon):    
    env = POMAPFEnv(config)
    env.load(grid_map, starts, goals)
    obs = env.observe()
    step = 0
    num_agents = len(starts)
    hidden = torch.zeros(num_agents, hidden_dim, dtype=float)
    paths = [[] for _ in range(num_agents)]
    while step <= horizon:
        curr_state = env.curr_state
        for i, loc in enumerate(curr_state):
            paths[i].append(tuple(loc))
        action, _, _ = model(torch.from_numpy(obs), hidden, curr_state)
        obs, reward, done, _, info = env.step(action)
        for i, loc in enumerate(curr_state):
            paths[i].append(tuple(loc))
        if all(done):
            break
        step += 1
    avg_step = 0.0
    for i in range(num_agents):
        while (len(paths[i]) > 1 and paths[i][-1] == paths[i][-2]):
            paths[i] = paths[i][:-1]
        avg_step += len(paths[i]) / num_agents
    return np.array_equal(env.curr_state, env.goal_state), avg_step, paths


def main(args):
    state_dict = torch.load(args.load_from_dir)
    model = AttentionPolicy(args.communication)
    model.load_state_dict(state_dict['policy']['target_policy'])
    num_instances = num_instances_per_test
    for map_name, num_agents in test_settings:
        file_name = f"./benchmarks/test_set/{map_name}_{num_agents}agents.pth"
        with open(file_name, 'rb') as f:
            instances = pickle.load(f)
        print(f"Testing instances for {map_name} with {num_agents} agents ...")
        success_rate, avg_step = 0.0, 0.0
        for grid_map, starts, goals in tqdm(instances[0: num_instances]):
            done, steps, paths = test_one_case(model, np.array(grid_map), list(starts), list(goals), max_timesteps[map_name])
            if done:
                success += 1 / num_instances
                avg_step += steps / num_instances
            else:
                avg_step += max_timesteps[map_name] / num_instances
        with open(f"./log/results.csv", 'a+') as f:
            height, width = np.shape(grid_map)
            num_obstacles = sum([row.count(OBSTACLE) for row in grid_map])
            method_name = 'SAHCA' if not args.communication else 'SACHA(C)'
            f.write(f"{method_name},{num_instances},{map_name},{height * width},{num_obstacles},{num_agents},{success_rate},{avg_step}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--communication", action='store_true')
    parser.add_argument("--load_from_dir", default="")
    args = parser.parse_args()
    main(args)