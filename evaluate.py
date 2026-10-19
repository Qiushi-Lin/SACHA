import os
import argparse
import pickle
import torch
import numpy as np
from tqdm import tqdm

from environment import POMAPFEnv
from model import SACHAActor
from utils import load_config, ROOT, OBSTACLE

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_one_case(env, grid_map, starts, goals, model, max_timestep, greedy=True):
    env.load(grid_map, starts, goals)
    obs, nbr, adj = env.observe()
    num_agents = len(starts)

    done = False
    model.reset()
    paths = [[tuple(loc)] for loc in env.agents_pos.tolist()]
    while not done and env.num_steps < max_timestep:
        actions = model.act(obs, nbr, adj, greedy=greedy)
        (obs, nbr, adj), _, done, _ = env.step(actions)
        for i, loc in enumerate(env.agents_pos.tolist()):
            paths[i].append(tuple(loc))
    flowtime = 0
    for i in range(num_agents):
        while len(paths[i]) > 1 and paths[i][-1] == paths[i][-2]:
            paths[i].pop()
        flowtime += len(paths[i]) - 1
    return done, flowtime


def main(args):
    checkpoint = torch.load(args.load_from_dir, map_location=device)
    config = load_config(args.config)
    model_config = dict(config)
    model_config.update({k: v for k, v in checkpoint['config'].items() if k not in ('test_settings', 'max_timesteps', 'num_instances_per_test')})
    model = SACHAActor(model_config)
    model.load_state_dict(checkpoint['actor'])
    model.to(device)
    model.eval()
    env = POMAPFEnv(model_config)

    num_instances = args.num_instances or config['num_instances_per_test']
    max_timesteps = config['max_timesteps']
    for map_name, num_agents in config['test_settings']:
        file_name = os.path.join(ROOT, 'benchmarks', 'test_set', f"{map_name}_{num_agents}agents.pth")
        with open(file_name, 'rb') as f:
            instances = pickle.load(f)
        instances = instances[:num_instances]
        print(f"Testing instances for {map_name} with {num_agents} agents ...")
        success = 0
        total_steps = 0.0
        for grid_map, starts, goals in tqdm(instances):
            done, flowtime = test_one_case(env, np.array(grid_map), np.array(starts), np.array(goals), model,
                                           max_timesteps[map_name], greedy=not args.sample)
            if done:
                success += 1
                total_steps += flowtime
            else:
                total_steps += num_agents * max_timesteps[map_name]
        success_rate = success / len(instances)
        avg_step = total_steps / (len(instances) * num_agents)
        print(f"success rate: {success_rate:.3f}, average step per agent: {avg_step:.2f}")
        with open(args.output, 'a+') as f:
            height, width = np.shape(grid_map)
            num_obstacles = int((np.array(grid_map) == OBSTACLE).sum())
            f.write(f"{args.method_name},{len(instances)},{map_name},{height * width},{num_obstacles},{num_agents}," +
                    f"{success_rate},{avg_step}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_from_dir", required=True, help='path to a checkpoint saved by train.py')
    parser.add_argument("--method_name", default="SACHA")
    parser.add_argument("--config", default=None)
    parser.add_argument("--num_instances", type=int, default=None)
    parser.add_argument("--sample", action='store_true', help='sample actions instead of taking the most likely one')
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()
    main(args)
