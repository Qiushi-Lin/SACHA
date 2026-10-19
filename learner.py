import os
from copy import deepcopy
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical

from model import SACHAActor, SACHACritic


class Learner:
    def __init__(self, config, buffer=None):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor = SACHAActor(config).to(self.device)
        self.critic = SACHACritic(config).to(self.device)
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)
        for p in list(self.actor_target.parameters()) + list(self.critic_target.parameters()):
            p.requires_grad_(False)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=config['lr_pi'])
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config['lr_q'])

        self.gamma = config['gamma']
        self.alpha = config['alpha']
        self.tau = config['tau']
        self.grad_clip = config['grad_clip']
        self.actor_loss_type = config['actor_loss']
        self.burn_in = config['burn_in']
        self.seq_len = config['seq_len']
        self.num_updates = config['num_updates']
        self.save_path = config['save_path']
        self.save_interval = config['save_interval']
        self.weights_publish_interval = config['weights_publish_interval']
        os.makedirs(self.save_path, exist_ok=True)

        self.buffer = buffer
        self.counter = 0
        self.last_counter = 0
        self.done = False
        self.stat_sums = {}
        self.stats_lock = threading.Lock()
        self.weights_lock = threading.Lock()
        self.publish_weights()

    def publish_weights(self):
        state_dict = {k: v.detach().cpu().clone() for k, v in self.actor.state_dict().items()}
        with self.weights_lock:
            self.weights = state_dict

    def get_weights(self):
        with self.weights_lock:
            return self.weights

    def save(self, name=None):
        path = os.path.join(self.save_path, '{}.pth'.format(self.counter if name is None else name))
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'config': self.config,
            'counter': self.counter,
        }, path)
        return path

    def unroll(self, actor, obs, nbr, adj, step_valid, start, end, grad):
        N = nbr.size(1)
        prev_nbr, prev_hidden = actor.initial_state(N, obs.device)
        logits_list, rep_list = [], []
        for t in range(end):
            with torch.set_grad_enabled(grad and t >= start):
                logits, rep, hidden = actor.forward_step(obs[t], nbr[t], adj[t], prev_nbr, prev_hidden)
            prev_nbr = torch.where(step_valid[t].unsqueeze(-1), nbr[t], torch.full_like(nbr[t], -1))
            prev_hidden = hidden
            if t >= start:
                logits_list.append(logits)
                rep_list.append(rep)
        return torch.stack(logits_list), torch.stack(rep_list)

    @staticmethod
    def flatten_nbr(nbr):
        S, N, _ = nbr.size()
        offset = (torch.arange(S, device=nbr.device) * N).view(S, 1, 1)
        return torch.where(nbr >= 0, nbr + offset, torch.full_like(nbr, -1)).view(S * N, -1)

    def update(self, batch):
        dev = self.device
        obs = torch.from_numpy(batch['obs']).to(dev).float()
        nbr = torch.from_numpy(batch['nbr']).to(dev)
        adj = torch.from_numpy(batch['adj']).to(dev)
        actions = torch.from_numpy(batch['actions']).to(dev)
        rewards = torch.from_numpy(batch['rewards']).to(dev)
        done = torch.from_numpy(batch['done']).to(dev)
        step_valid = torch.from_numpy(batch['step_valid']).to(dev)
        train_mask = torch.from_numpy(batch['train_mask']).to(dev)
        weight = torch.from_numpy(batch['weight']).to(dev)

        B, S = self.burn_in, self.seq_len
        N = nbr.size(1)
        A = self.actor.action_dim
        cur = slice(B, B + S)
        nxt = slice(B + 1, B + S + 1)

        logits, rep = self.unroll(self.actor, obs, nbr, adj, step_valid, B, B + S, grad=True)
        with torch.no_grad():
            target_logits, target_rep = self.unroll(self.actor_target, obs, nbr, adj, step_valid, B + 1, B + S + 1, grad=False)

        logits = logits.reshape(S * N, A)
        rep = rep.reshape(S * N, -1)
        nbr_cur = self.flatten_nbr(nbr[cur])
        act_cur = actions[cur].reshape(-1)
        w = (train_mask[cur] * weight.unsqueeze(0)).reshape(-1)
        w_sum = w.sum().clamp(min=1e-8)

        q_all = self.critic(rep.detach(), nbr_cur, act_cur)
        q_taken = q_all.gather(1, act_cur.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target_logits = target_logits.reshape(S * N, A)
            next_log_pi = F.log_softmax(target_logits, dim=-1)
            next_pi = next_log_pi.exp()
            next_actions = Categorical(probs=next_pi).sample()
            next_q = self.critic_target(target_rep.reshape(S * N, -1), self.flatten_nbr(nbr[nxt]), next_actions)
            next_v = (next_pi * (next_q - self.alpha * next_log_pi)).sum(-1)
            y = rewards[cur].reshape(-1) + self.gamma * (1 - done[cur].reshape(-1)) * next_v
        td_error = q_taken - y
        critic_loss = (w * td_error ** 2).sum() / w_sum

        log_pi = F.log_softmax(logits, dim=-1)
        pi = log_pi.exp()
        q_det = q_all.detach()
        if self.actor_loss_type == 'score':
            sampled = Categorical(logits=logits.detach()).sample()
            log_pi_a = log_pi.gather(1, sampled.unsqueeze(1)).squeeze(1)
            baseline = (pi.detach() * q_det).sum(-1)
            advantage = q_det.gather(1, sampled.unsqueeze(1)).squeeze(1) - baseline - self.alpha * log_pi_a.detach()
            actor_loss = -(w * log_pi_a * advantage).sum() / w_sum
        elif self.actor_loss_type == 'expected':
            actor_loss = (w * (pi * (self.alpha * log_pi - q_det)).sum(-1)).sum() / w_sum
        else:
            raise ValueError('unknown actor_loss {}'.format(self.actor_loss_type))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)

        with torch.no_grad():
            entropy = (w * -(pi * log_pi).sum(-1)).sum() / w_sum
            mean_q = (w * q_taken).sum() / w_sum
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'entropy': entropy.item(),
            'q': mean_q.item(),
            'batch_agents': N,
        }

    @torch.no_grad()
    def soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.lerp_(sp, self.tau)

    def step(self, batch):
        stats = self.update(batch)
        with self.stats_lock:
            for k, v in stats.items():
                self.stat_sums[k] = self.stat_sums.get(k, 0.0) + v
            self.counter += 1
        if self.counter % self.weights_publish_interval == 0:
            self.publish_weights()
        if self.counter % self.save_interval == 0:
            self.save()
        return stats

    def run(self):
        self.learning_thread = threading.Thread(target=self.train, daemon=True)
        self.learning_thread.start()

    def train(self):
        import ray
        try:
            while self.counter < self.num_updates:
                batch = ray.get(self.buffer.get_data.remote())
                self.step(batch)
                if self.counter % 100 == 0 and ray.get(self.buffer.curriculum_done.remote()):
                    print('curriculum finished')
                    break
            self.save('final')
        finally:
            self.done = True

    def stats(self, interval: int):
        with self.stats_lock:
            num = self.counter - self.last_counter
            out = {'updates': self.counter, 'update_speed': num / interval}
            if num > 0:
                for k, v in self.stat_sums.items():
                    out[k] = v / num
            self.stat_sums = {}
            self.last_counter = self.counter
        return self.done, out
