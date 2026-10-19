import os
import argparse
import random
import time
import torch
import numpy as np

from utils import load_config, ROOT
from buffer import ReplayBuffer
from worker import Worker
from learner import Learner

os.environ["OMP_NUM_THREADS"] = "1"


def print_learner_stats(stats):
    print('number of updates: {}, update speed: {:.2f}/s'.format(stats['updates'], stats['update_speed']))
    if 'critic_loss' in stats:
        print('critic loss: {:.4f}, actor loss: {:.4f}, entropy: {:.3f}, Q: {:.3f}, agents per batch: {:.0f}'.format(
            stats['critic_loss'], stats['actor_loss'], stats['entropy'], stats['q'], stats['batch_agents']))


def train_single(config, args):
    buffer = ReplayBuffer(config)
    learner = Learner(config)
    worker = Worker(0, config, seed=args.seed)
    worker.set_weights(learner.get_weights())
    print('device: {}'.format(learner.device))

    episodes, steps, last_log = 0, 0, time.time()
    while learner.counter < config['num_updates']:
        episode = worker.run_episode()
        buffer.add(episode)
        episodes += 1
        steps += episode['length']
        if buffer.ready():
            for _ in range(args.updates_per_episode):
                learner.step(buffer.sample())
                if learner.counter >= config['num_updates']:
                    break
            worker.set_weights(learner.get_weights())
        worker.set_env_setting_set(buffer.get_env_setting_set())

        if time.time() - last_log >= config['log_interval']:
            interval = time.time() - last_log
            print('episodes: {}, env steps: {}'.format(episodes, steps))
            _, stats = learner.stats(interval)
            print_learner_stats(stats)
            buffer.stats(interval)
            print()
            last_log = time.time()

        if buffer.curriculum_done():
            print('curriculum finished')
            break

    print('saved final model to {}'.format(learner.save('final')))


def train_ray(config, args):
    import ray
    ray.init(runtime_env={'env_vars': {'PYTHONPATH': ROOT, 'OMP_NUM_THREADS': '1'}})
    num_gpus = 1 if torch.cuda.is_available() else 0
    num_cpus = int(ray.available_resources().get('CPU', 1))
    num_workers = max(1, min(config['num_workers'], num_cpus - 2))
    if num_workers < config['num_workers']:
        print('only {} CPUs available, using {} workers'.format(num_cpus, num_workers))

    RemoteBuffer = ray.remote(num_cpus=1)(ReplayBuffer)
    RemoteLearner = ray.remote(num_cpus=1, num_gpus=num_gpus)(Learner)
    RemoteWorker = ray.remote(num_cpus=1)(Worker)

    buffer = RemoteBuffer.remote(config)
    learner = RemoteLearner.remote(config, buffer)
    workers = [RemoteWorker.remote(i, config, learner, buffer, args.seed + i) for i in range(num_workers)]
    for worker in workers:
        worker.run.remote()

    while not ray.get(buffer.ready.remote()):
        time.sleep(5)
        ray.get(buffer.stats.remote(5))
        print()
    print('Start training...')
    buffer.run.remote()
    learner.run.remote()
    done = False
    while not done:
        time.sleep(config['log_interval'])
        done, stats = ray.get(learner.stats.remote(config['log_interval']))
        print_learner_stats(stats)
        ray.get(buffer.stats.remote(config['log_interval']))
        print()
    ray.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, help='path to the config file (default: ./config.yaml)')
    parser.add_argument('--single', action='store_true', help='run everything in one process without ray')
    parser.add_argument('--updates_per_episode', type=int, default=4, help='learner updates per collected episode (single mode)')
    parser.add_argument('--use_comm', action='store_true', help='train SACHA(C)')
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--num_updates', type=int, default=None)
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.use_comm:
        config['use_comm'] = True
    for key in ['num_workers', 'num_updates', 'save_path']:
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.single:
        train_single(config, args)
    else:
        train_ray(config, args)


if __name__ == '__main__':
    main()
