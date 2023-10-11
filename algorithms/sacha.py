import copy
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import MSELoss

from utils.misc import soft_update, enable_gradients, disable_gradients
from model import AttentionPolicy, AttentionCritic


class SACHA:
    def __init__(self, config, communication, device):
        self.device = device
        self.policy = AttentionPolicy(communication)
        self.target_policy = copy.deepcopy(self.policy)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=config['lr_pi'])
        self.critic = AttentionCritic()
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config['lr_q'])
        self.tau = config['tau']
        self.gamma = config['gamma']
        self.alpha = config['alpha']
    
    def step(self, obs, hidden, state):
        return self.policy(obs, hidden, state)

    def update_policy(self, samples, soft=True):
        b_obs, b_action, b_reward, b_next_obs, b_done, b_info, b_hidden, b_state = samples
        num_samples = len(b_state)
        num_agents = len(b_state[0])
        loss = 0.0
        for i in range(num_samples):
            action, _, log_pi = self.policy(b_obs[i], b_hidden[i], b_state[i])
            q_n = self.critic(b_obs[i], action, b_state[i])
            b_n = self.critic.get_coma_baseline(b_obs[i], action, b_state[i])
            for j in range(num_agents):
                a = action[j]
                if soft:
                    loss += log_pi[j][a] * (log_pi[j][a] * self.alpha - q_n[j][a] + b_n[j][a])
                else:
                    loss += log_pi[j][a] * (- q_n[j][a] + b_n[j][a])
        loss /= (num_agents * num_samples)
        disable_gradients(self.critic)
        loss.backward()
        enable_gradients(self.critic)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.policy_optimizer.step()
        self.policy_optimizer.zero_grad()  

    def update_critic(self, samples, soft=True):
        b_obs, b_action, b_reward, b_next_obs, b_done, b_info, b_hidden, b_state = samples
        num_samples = len(b_state)
        num_agents = len(b_state[0])
        loss = 0.0
        for i in range(num_samples):
            next_action, next_hidden, next_log_pi = self.target_policy(b_next_obs[i], b_hidden[i], b_state[i])
            q_n = self.critic(b_obs[i], b_action[i], b_state[i])
            next_q_n = self.critic(b_next_obs[i], next_action, b_state[i])
            target_q_n = []
            for j in range(num_agents):
                next_a = next_action[j]
                target_q = b_reward[i][j] + self.gamma * next_q_n[j][next_a] * (1 - b_done[i][j])
                if soft:
                    target_q -= self.alpha * next_log_pi[j][next_a]
                target_q_n.append(target_q)
            q_n = torch.stack([q_n[j][int(b_action[i][j])] for j in range(num_agents)])
            target_q_n = torch.stack(target_q_n)
            loss += MSELoss()(q_n, target_q_n)
        loss /= (num_samples * num_agents)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

    def update_targets(self):
        soft_update(self.target_policy, self.policy, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)
    
    def prep_training(self, device):
        self.policy.train()
        self.critic.train()
        self.target_policy.train()
        self.target_critic.train()
        self.policy.to(device)
        self.critic.to(device)
        self.target_policy.to(device)
        self.target_critic.to(device)

    def prep_rollout(self, device):
        self.policy.eval()
        self.policy.to(device)

    def save(self, filename):
        state_dict = {
            'policy' : {
                'policy': self.policy.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict()
            },
            'critic' : {
                'critic': self.critic.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()
            }
        }
        torch.save(state_dict, filename)
