import os
from copy import deepcopy
import threading
import ray
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda.amp import GradScaler
import numpy as np
from model import PolicyNet
from buffer import GlobalBuffer

# config
import yaml
config = yaml.safe_load(open("./config.yaml", 'r'))


@ray.remote(num_cpus=1, num_gpus=1)
class Learner:
    def __init__(self, buffer: GlobalBuffer):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PolicyNet()
        self.model.to(self.device)
        self.target_model = deepcopy(self.model)
        self.optimizer = Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = MultiStepLR(self.optimizer, milestones=[200000, 400000], gamma=0.5)
        self.buffer = buffer
        self.counter = 0
        self.last_counter = 0
        self.done = False
        self.loss = 0
        self.num_episodes = config['num_episodes']
        self.forward_steps = config['forward_steps']
        self.episodes_per_target_update = config['episodes_per_target_update']
        self.save_path = config['save_path']
        self.store_weights()

    def get_weights(self):
        return self.weights_id

    def store_weights(self):
        state_dict = self.model.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()
        self.weights_id = ray.put(state_dict)

    def run(self):
        self.learning_thread = threading.Thread(target=self.train, daemon=True)
        self.learning_thread.start()

    def train(self):
        scaler = GradScaler()
        while not ray.get(self.buffer.check_done.remote()) and self.counter < self.num_episodes:
            for i in range(1, 10001):
                data_id = ray.get(self.buffer.get_data.remote())
                data = ray.get(data_id)
    
                b_obs, b_action, b_reward, b_done, b_steps, b_seq_len, b_hidden, b_comm_mask, indices, weights, old_ptr = data
                b_obs, b_action, b_reward = b_obs.to(self.device), b_action.to(self.device), b_reward.to(self.device)
                b_done, b_steps, weights = b_done.to(self.device), b_steps.to(self.device), weights.to(self.device)
                b_hidden = b_hidden.to(self.device)
                b_comm_mask = b_comm_mask.to(self.device)

                b_next_seq_len = [ (seq_len+forward_steps).item() for seq_len, forward_steps in zip(b_seq_len, b_steps) ]
                b_next_seq_len = torch.LongTensor(b_next_seq_len)

                with torch.no_grad():
                    b_q_ = (1 - b_done) * self.target_model(b_obs, b_next_seq_len, b_hidden, b_comm_mask).max(1, keepdim=True)[0]

                b_q = self.model(b_obs[:, :-self.forward_steps], b_seq_len, b_hidden, b_comm_mask[:, :-self.forward_steps]).gather(1, b_action)

                td_error = (b_q - (b_reward + (0.99 ** b_steps) * b_q_))

                priorities = td_error.detach().squeeze().abs().clamp(1e-4).cpu().numpy()

                loss = (weights * self.huber_loss(td_error)).mean()
                self.loss += loss.item()

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()

                scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 40)

                scaler.step(self.optimizer)
                scaler.update()
 
                self.scheduler.step()

                # store new weights in shared memory
                if i % 5  == 0:
                    self.store_weights()

                self.buffer.update_priorities.remote(indices, priorities, old_ptr)

                self.counter += 1

                # update target net, save model
                if i % self.episodes_per_target_update == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
                
                if i % self.save_interval == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.save_path, '{}.pth'.format(self.counter)))

        self.done = True

    def huber_loss(self, td_error, kappa=1.0):
        abs_td_error = td_error.abs()
        flag = (abs_td_error < kappa).float()
        return flag * abs_td_error.pow(2) * 0.5 + (1 - flag) * (abs_td_error - 0.5)

    def stats(self, interval: int):
        print('number of updates: {}'.format(self.counter))
        print('update speed: {}/s'.format((self.counter-self.last_counter)/interval))
        if self.counter != self.last_counter:
            print('loss: {:.4f}'.format(self.loss / (self.counter-self.last_counter)))
        self.last_counter = self.counter
        self.loss = 0
        return self.done