import os
import random
import time
import torch
import numpy as np
import ray

from buffer import GlobalBuffer
from worker import Worker
from learner import Learner

os.environ["OMP_NUM_THREADS"] = "1"
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# config
import yaml
config = yaml.safe_load(open("./config.yaml", 'r'))
num_workers = config['num_workers']
log_interval = config['log_interval']


def main():
    ray.init()
    buffer = GlobalBuffer.remote()
    learner = Learner.remote(buffer)
    time.sleep(1)
    workers = [Worker.remote(i, 0.4**(1+(i/(num_workers-1))*7), learner, buffer) for i in range(num_workers)]
    for worker in workers:
        worker.run.remote()
    while not ray.get(buffer.ready.remote()):
        time.sleep(5)
        ray.get(learner.stats.remote(5))
        ray.get(buffer.stats.remote(5))
    print('Start training...')
    buffer.run.remote()
    learner.run.remote()
    done = False
    while not done:
        time.sleep(log_interval)
        done = ray.get(learner.stats.remote(log_interval))
        ray.get(buffer.stats.remote(log_interval))
        print()


if __name__ == '__main__':
    main()