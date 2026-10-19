import random
import threading
import time
from collections import deque
import numpy as np

from utils import build_observations, NUM_OBS_CHANNELS
from environment import setting_feasible

UNREACHABLE = np.iinfo(np.uint16).max


def compress_dist(padded_dist):
    finite = np.isfinite(padded_dist)
    out = np.full(padded_dist.shape, UNREACHABLE, dtype=np.uint16)
    out[finite] = np.minimum(padded_dist[finite], UNREACHABLE - 1)
    return out


def decompress_dist(compressed):
    out = compressed.astype(np.float32)
    out[compressed == UNREACHABLE] = np.inf
    return out


class Curriculum:
    def __init__(self, config):
        self.window = config['curriculum_window']
        self.upgrade_rate = config['upgrade_rate']
        self.max_num_agents = config['max_num_agents']
        self.max_map_length = config['max_map_length']
        self.agent_increment = config['agent_increment']
        self.map_increment = config['map_increment']
        self.stats = {tuple(config['default_env_setting']): deque(maxlen=self.window)}
        self.passed = set()

    def record(self, setting, success):
        if setting not in self.stats:
            return
        self.stats[setting].append(bool(success))
        if setting not in self.passed and self.success_rate(setting) is not None \
                and self.success_rate(setting) >= self.upgrade_rate:
            self.passed.add(setting)
            num_agents, map_length = setting
            for candidate in [(num_agents + self.agent_increment, map_length), (num_agents, map_length + self.map_increment)]:
                if candidate[0] <= self.max_num_agents and candidate[1] <= self.max_map_length \
                        and setting_feasible(*candidate) and candidate not in self.stats:
                    self.stats[candidate] = deque(maxlen=self.window)

    def success_rate(self, setting):
        s = self.stats[setting]
        if len(s) < self.window:
            return None
        return sum(s) / len(s)

    def settings(self):
        return list(self.stats.keys())

    def done(self):
        return any(n >= self.max_num_agents and l >= self.max_map_length for n, l in self.passed)


class ReplayBuffer:
    def __init__(self, config):
        self.config = config
        self.capacity = config['buffer_capacity']
        self.learning_threshold = config['learning_threshold']
        self.batch_agents = config['batch_agents']
        self.max_batch_sequences = config['max_batch_sequences']
        self.seq_len = config['seq_len']
        self.burn_in = config['burn_in']
        self.K = config['K_obs']
        self.obs_radius = config['obs_radius']
        self.heuristic_norm = config['heuristic_norm']
        self.prefetch_batches = config['prefetch_batches']

        self.episodes = []
        self.ptr = 0
        self.num_transitions = 0
        self.counter = 0
        self.lock = threading.Lock()
        self.curriculum = Curriculum(config)
        self.batched_data = deque()

    def __len__(self):
        return self.num_transitions

    def add(self, episode: dict):
        with self.lock:
            if len(self.episodes) < self.capacity:
                self.episodes.append(episode)
            else:
                self.num_transitions -= self.episodes[self.ptr]['length']
                self.episodes[self.ptr] = episode
                self.ptr = (self.ptr + 1) % self.capacity
            self.num_transitions += episode['length']
            self.counter += episode['length']
            self.curriculum.record((episode['num_agents'], episode['map_len']), episode['done'])

    def ready(self):
        return self.num_transitions >= self.learning_threshold

    def sample(self):
        with self.lock:
            episodes = list(self.episodes)
        lengths = np.array([e['length'] for e in episodes], dtype=np.float64)
        probs = lengths / lengths.sum()
        chosen, total_agents = [], 0
        while total_agents < self.batch_agents and len(chosen) < self.max_batch_sequences:
            episode = episodes[np.random.choice(len(episodes), p=probs)]
            chosen.append((episode, random.randrange(episode['length'])))
            total_agents += episode['num_agents']
        return self.collate(chosen)

    def collate(self, chosen):
        T = self.burn_in + self.seq_len + 1
        N = sum(episode['num_agents'] for episode, _ in chosen)
        L = 2 * self.obs_radius + 1

        obs = np.zeros((T, N, self.K, NUM_OBS_CHANNELS, L, L), dtype=np.float16)
        nbr = np.full((T, N, self.K), -1, dtype=np.int64)
        nbr[:, :, 0] = np.arange(N)
        adj = np.zeros((T, N, N), dtype=bool)
        actions = np.zeros((T, N), dtype=np.int64)
        rewards = np.zeros((T, N), dtype=np.float32)
        done = np.zeros((T, N), dtype=np.float32)
        step_valid = np.zeros((T, N), dtype=bool)
        train_mask = np.zeros((T, N), dtype=np.float32)
        weight = np.zeros(N, dtype=np.float32)

        offset = 0
        for episode, t0 in chosen:
            M, length = episode['num_agents'], episode['length']
            agents = slice(offset, offset + M)
            taus = t0 - self.burn_in + np.arange(T)

            has_state = (taus >= 0) & (taus <= length)
            idx = np.flatnonzero(has_state)
            o, nb, ad = build_observations(episode['padded_map'], decompress_dist(episode['padded_dist']), episode['dmax'],
                                           episode['pos'][taus[idx]], self.K, self.obs_radius, self.heuristic_norm)
            obs[idx, agents] = o
            nbr[idx, agents] = np.where(nb >= 0, nb + offset, -1)
            adj[idx, agents, agents] = ad
            step_valid[idx, agents] = True

            idx = np.flatnonzero((taus >= 0) & (taus < length))
            actions[idx, agents] = episode['actions'][taus[idx]]
            rewards[idx, agents] = episode['rewards'][taus[idx]]
            if episode['done']:
                done[np.flatnonzero(taus == length - 1), agents] = 1
            train = (np.arange(T) >= self.burn_in) & (np.arange(T) < self.burn_in + self.seq_len) & (taus < length)
            train_mask[np.flatnonzero(train), agents] = 1

            weight[agents] = 1.0 / M
            offset += M

        return {
            'obs': obs, 'nbr': nbr, 'adj': adj, 'actions': actions, 'rewards': rewards, 'done': done,
            'step_valid': step_valid, 'train_mask': train_mask, 'weight': weight,
        }

    def run(self):
        self.background_thread = threading.Thread(target=self.prepare_data, daemon=True)
        self.background_thread.start()

    def prepare_data(self):
        while True:
            if len(self.batched_data) < self.prefetch_batches:
                self.batched_data.append(self.sample())
            else:
                time.sleep(0.01)

    def get_data(self):
        if len(self.batched_data) == 0:
            return self.sample()
        return self.batched_data.popleft()

    def get_env_setting_set(self):
        return self.curriculum.settings()

    def curriculum_done(self):
        return self.curriculum.done()

    def stats(self, interval: int):
        print('buffer update speed: {:.1f} steps/s'.format(self.counter / interval))
        print('buffer size: {} steps in {} episodes'.format(self.num_transitions, len(self.episodes)))
        settings = self.curriculum.settings()
        print('curriculum: {} settings, {} passed'.format(len(settings), len(self.curriculum.passed)))
        for setting in settings:
            if setting in self.curriculum.passed:
                continue
            s = self.curriculum.stats[setting]
            rate = sum(s) / len(s) if len(s) > 0 else 0.0
            print('  agents {:3d}, map {:3d}x{:<3d}: success {:.2f} ({} episodes)'.format(setting[0], setting[1], setting[1], rate, len(s)))
        self.counter = 0
