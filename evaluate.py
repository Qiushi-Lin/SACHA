import argparse
import torch
import numpy as np
from tqdm import tqdm
import pickle

from environment import POMAPFEnv
from model import AttentionPolicy

# config
import yaml
config = yaml.safe_load(open("./config.yaml", 'r'))
OBSTACLE, FREE_SPACE = config['grid_map']['OBSTACLE'], config['grid_map']['FREE_SPACE']
num_instances_per_test = config['num_instances_per_test']
test_settings = config['test_settings']
max_timesteps = config['max_timesteps']

# device (CUDA_VISIBLE_DEVICES=GPU_ID)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
rollout_device = 'cpu'


def test_one_case(grid_map, starts, goals, model, max_timestep):
    env = POMAPFEnv()
    env.load(grid_map, starts, goals)
    obs, pos = env.observe()
    num_agents = len(starts)
    
    done = False
    model.reset()
    paths = [[] for _ in range(num_agents)]
    for i, loc in enumerate(pos):
        paths[i].append(tuple(loc))
    while not done and env.steps < max_timestep:
        actions, _, _, _ = model.step(torch.as_tensor(obs.astype(np.float32)), torch.as_tensor(pos.astype(np.float32)))
        (obs, pos), _, done, _ = env.step(actions)
        for i, loc in enumerate(pos):
            paths[i].append(tuple(loc))
    flowtime = 0
    for i in range(num_agents):
        while len(paths[i]) > 1 and paths[i][-1] == paths[i][-2]:
            paths[i].pop()
        flowtime += len(paths[i])
    return np.array_equal(env.agents_pos, env.goals_pos), flowtime


def main(args):
    for map_name, num_agents in test_settings:
        file_name = f"./benchmarks/test_set/{map_name}_{num_agents}agents.pth"
        with open(file_name, 'rb') as f:
            instances = pickle.load(f)
        print(f"Testing instances for {map_name} with {num_agents} agents ...")
        success = 0
        avg_flowtime = 0.0
        for grid_map, starts, goals in tqdm(instances[0: num_instances_per_test]):
            model = AttentionPolicy()
            model.to(device)
            state_dict = torch.load(args.load_from_dir, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            model.share_memory()
            done, flowtime = test_one_case(np.array(grid_map), np.array(starts), np.array(goals), model, max_timesteps[map_name])
            if done:
                success += 1
                avg_flowtime += flowtime
            else:
                avg_flowtime += num_agents * max_timesteps[map_name]
        with open(f"results.csv", 'a+') as f:
            height, width = np.shape(grid_map)
            num_obstacles = sum([row.count(OBSTACLE) for row in grid_map])
            f.write(f"{args.method_name},{num_instances_per_test},{map_name},{height * width},{num_obstacles},{num_agents}," +\
                f"{success / num_instances_per_test},{avg_flowtime / (num_instances_per_test * num_agents)}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_from_dir", default="")
    parser.add_argument("--method_name", default="POMAPF")
    args = parser.parse_args()
    main(args)