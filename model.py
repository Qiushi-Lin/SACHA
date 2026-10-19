import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from utils import NUM_OBS_CHANNELS


class ResBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.block1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.block2 = nn.Conv2d(channel, channel, 3, 1, 1)

    def forward(self, x):
        identity = x
        x = self.block1(x)
        x = F.relu(x)
        x = self.block2(x)
        x = x + identity
        x = F.relu(x)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, mask):
        N, K, D = x.size()
        q, k, v = self.qkv(x).view(N, K, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~mask[:, None, None, :], float('-inf'))
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).reshape(N, K, D)
        return self.out(context)


class GCNCommBlock(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_layers)])

    def forward(self, x, adj):
        a = adj.float() + torch.eye(x.size(0), device=x.device)
        d = a.sum(1).rsqrt()
        a = d[:, None] * a * d[None, :]
        for layer in self.layers:
            x = torch.sigmoid(a @ layer(x))
        return x


def carry_hidden(nbr, prev_nbr, prev_hidden):
    match = (nbr.unsqueeze(2) == prev_nbr.unsqueeze(1)) & (nbr >= 0).unsqueeze(2)
    return torch.bmm(match.to(prev_hidden.dtype), prev_hidden)


class SACHAActor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.obs_radius = config['obs_radius']
        self.fov = 2 * self.obs_radius + 1
        self.K = config['K_obs']
        self.hidden_dim = config['hidden_dim']
        self.action_dim = config['action_dim']
        num_channels = config['num_channels']

        self.obs_encoder = nn.Sequential(
            nn.Conv2d(NUM_OBS_CHANNELS, num_channels, 3, 1),
            nn.ReLU(),
            ResBlock(num_channels),
            ResBlock(num_channels),
            ResBlock(num_channels),
            nn.Conv2d(num_channels, 16, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.latent_dim = 16 * (self.fov - 2) ** 2
        self.recurrent = nn.GRUCell(self.latent_dim, self.hidden_dim)
        self.attention = MultiHeadAttention(self.hidden_dim, config['num_attn_heads'])
        self.comm = GCNCommBlock(self.hidden_dim, config['num_gcn_layers']) if config['use_comm'] else None
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.reset()

    def initial_state(self, num_agents, device):
        prev_nbr = torch.full((num_agents, self.K), -1, dtype=torch.long, device=device)
        prev_hidden = torch.zeros((num_agents, self.K, self.hidden_dim), device=device)
        return prev_nbr, prev_hidden

    def forward_step(self, obs, nbr, adj, prev_nbr, prev_hidden):
        valid = nbr >= 0
        hidden_in = carry_hidden(nbr, prev_nbr, prev_hidden)

        latent = self.obs_encoder(obs[valid])
        hidden = torch.zeros_like(hidden_in)
        hidden[valid] = self.recurrent(latent, hidden_in[valid])

        context = self.attention(hidden, valid)
        rep = (context * valid.unsqueeze(-1)).sum(1)
        if self.comm is not None:
            rep = self.comm(rep, adj)
        logits = self.decoder(rep)
        return logits, rep, hidden

    def reset(self):
        self.prev_nbr = None
        self.prev_hidden = None

    @torch.no_grad()
    def act(self, obs, nbr, adj, greedy=False):
        device = next(self.parameters()).device
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        nbr = torch.as_tensor(nbr, dtype=torch.long, device=device)
        adj = torch.as_tensor(adj, dtype=torch.bool, device=device)
        if self.prev_nbr is None or self.prev_nbr.size(0) != nbr.size(0):
            self.prev_nbr, self.prev_hidden = self.initial_state(nbr.size(0), device)
        logits, _, hidden = self.forward_step(obs, nbr, adj, self.prev_nbr, self.prev_hidden)
        self.prev_nbr, self.prev_hidden = nbr, hidden
        if greedy:
            actions = logits.argmax(-1)
        else:
            actions = Categorical(logits=logits).sample()
        return actions.cpu().numpy()


class SACHACritic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config['hidden_dim']
        self.action_dim = config['action_dim']
        self.encoder = nn.Sequential(
            nn.Linear(self.hidden_dim + self.action_dim + 1, self.hidden_dim),
            nn.ReLU(),
        )
        self.attention = MultiHeadAttention(self.hidden_dim, config['num_attn_heads'])
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

    def forward(self, rep, nbr, actions):
        N, K = nbr.size()
        valid = nbr >= 0
        idx = nbr.clamp(min=0)
        members = rep[idx]
        member_actions = F.one_hot(actions[idx], self.action_dim).float()
        member_actions[:, 0] = 0
        member_actions = member_actions * valid.unsqueeze(-1)
        is_self = torch.zeros((N, K, 1), device=rep.device)
        is_self[:, 0] = 1
        x = self.encoder(torch.cat([members, member_actions, is_self], dim=-1))
        context = self.attention(x, valid)
        context = (context * valid.unsqueeze(-1)).sum(1)
        return self.decoder(context)
