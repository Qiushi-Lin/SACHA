import argparse
import os
import numpy as np
import torch
from torch.autograd import Variable
import wandb

from environment import POMAPFEnv
from utils.replay_buffer import ReplayBuffer
from algorithms.sacha import SACHA
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv


# device (CUDA_VISIBLE_DEVICES=GPU_ID)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
rollout_device = 'cpu'

# config
import yaml
config_base = yaml.safe_load(open("./configs/config_base.yaml", 'r'))
config_train = yaml.safe_load(open("./configs/config_train.yaml", 'r'))
config = config_base | config_train
OBSTACLE, FREE_SPACE = config['grid_map']['OBSTACLE'], config['grid_map']['FREE_SPACE']
max_num_agents = config['max_num_agents']
max_map_size = config['max_map_size']
num_episodes = config['num_episodes']
num_rollout_threads = config['num_rollout_threads']
hidden_dim = config['hidden_dim']
batch_size = config['batch_size']
steps_per_update = config['steps_per_update']
upgrade_threshold = config['upgrade_threshold']
curriculum = config['curriculum']


def make_parallel_env(num_envs):
    def get_env_fn(seed):
        def init_env():
            env = POMAPFEnv(config)
            env.seed(seed)
            return env
        return init_env
    return SubprocVecEnv([get_env_fn(i) for i in range(num_envs)])


def main(args):
    log_dir = './log/SACHA' if not args.communication else './log/SACHA(C)'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    num_agents = config['default_num_agents']
    map_size = config['default_map_size']
    env = make_parallel_env(num_rollout_threads)
    algo = SACHA(config, args.communication, device)
    buffer = ReplayBuffer(config, num_agents)
    for e in range(0, num_episodes, num_rollout_threads):
        b_obs = env.reset()
        b_state = env.env_method('get_curr_state')
        b_hidden = torch.zeros(num_rollout_threads, num_agents, hidden_dim, dtype=float)
        b_next_hidden = torch.zeros(num_rollout_threads, num_agents, hidden_dim, dtype=float)
        algo.prep_rollout(rollout_device)
        for t in range(config['max_episode_length']):
            b_action = []
            for i in range(num_rollout_threads):
                obs, hidden, state = torch.from_numpy(b_obs[i]), b_hidden[i], torch.from_numpy(np.array(b_state[i]))
                action, hidden, _ = algo.step(obs, hidden, state)
                b_action.append(torch.stack(action).detach().numpy())
                b_next_hidden[i] = hidden
            b_next_obs, b_reward, b_done, b_info = env.step(b_action)
            buffer.add(b_obs, b_action, b_reward, b_next_obs, b_done, b_info, b_hidden, b_state)
            b_hidden = b_next_hidden
            if len(buffer) >= batch_size and (t+1) % steps_per_update == 0:
                algo.prep_training(device)
                samples = buffer.sample(batch_size, device)
                algo.update_policy(samples)
                algo.update_critic(samples)
                algo.update_targets()
                algo.prep_rollout(rollout_device)
        if (e+1) % config['save_interval'] == 0:
            algo.prep_rollout(rollout_device)            
            algo.save(os.path.join(log_dir, f"model_ep{e+1}.pth"))
            print(f"episode {e+1} finished and model_ep{e+1} saved")
        if curriculum and buffer.get_succes_rate() >= upgrade_threshold:
            if num_agents + 4 < max_num_agents and map_size + 10 < max_map_size:
                if np.random.uniform() < 0.5:
                    num_agents += 4
                else:
                    map_size += 10
            else:
                if num_agents + 4 < max_num_agents:
                    num_agents += 4
                elif map_size + 10 < max_map_size:
                    map_size += 10
            env.env_method('set_level', map_size, num_agents)
            buffer = ReplayBuffer(config, num_agents)
    env.close()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--communication", action='store_true')
    args = parser.parse_args()
    # wandb.init(project='SACHA', config=config_train)
    if not args.communication:
        print("Training SACHA...")
    else:
        print("Training SACHA(C)...")
    main(args)