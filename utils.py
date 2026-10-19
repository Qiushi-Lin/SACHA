import os
import random
from collections import deque
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_config(path=None):
    if path is None:
        path = os.path.join(ROOT, 'config.yaml')
    with open(path, 'r') as f:
        return yaml.safe_load(f)


config = load_config()
FREE_SPACE, OBSTACLE = config['grid_map']['FREE_SPACE'], config['grid_map']['OBSTACLE']
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NUM_OBS_CHANNELS = 2 + len(DIRECTIONS) + 2


def generate_random_map(height, width, num_obstacles):
    grid_map = [[FREE_SPACE for _ in range(width)] for _ in range(height)]
    counter = 0
    while counter < num_obstacles:
        i = random.randint(0, height - 1)
        j = random.randint(0, width  - 1)
        if grid_map[i][j] == FREE_SPACE:
            grid_map[i][j] = OBSTACLE
            counter += 1
    return grid_map


def map_partition(grid_map):
    grid = np.asarray(grid_map)
    height, width = grid.shape
    free = grid == FREE_SPACE
    visited = np.zeros_like(free)
    partitions = []
    for x, y in np.argwhere(free).tolist():
        if visited[x, y]:
            continue
        visited[x, y] = True
        component = []
        open_list = deque([(x, y)])
        while open_list:
            cx, cy = open_list.popleft()
            component.append((cx, cy))
            for dx, dy in DIRECTIONS:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < height and 0 <= ny < width and free[nx, ny] and not visited[nx, ny]:
                    visited[nx, ny] = True
                    open_list.append((nx, ny))
        partitions.append(component)
    return partitions


def sample_starts_goals(partitions, num_agents):
    parts = [list(p) for p in partitions if len(p) >= 2]
    assert sum(len(p) // 2 for p in parts) >= num_agents, 'not enough free cells'
    starts, goals = [], []
    for _ in range(num_agents):
        sizes = np.array([len(p) for p in parts], dtype=np.float64)
        idx = np.random.choice(len(parts), p=sizes / sizes.sum())
        part = parts[idx]
        for locs in (starts, goals):
            j = random.randrange(len(part))
            part[j], part[-1] = part[-1], part[j]
            locs.append(part.pop())
        if len(part) < 2:
            parts.pop(idx)
    return np.array(starts, dtype=int), np.array(goals, dtype=int)


def bfs_distance_maps(free, goals):
    num_agents = len(goals)
    height, width = free.shape
    dist = np.full((num_agents, height, width), np.inf, dtype=np.float32)
    frontier = np.zeros((num_agents, height, width), dtype=bool)
    frontier[np.arange(num_agents), goals[:, 0], goals[:, 1]] = True
    dist[frontier] = 0
    visited = frontier.copy()
    d = 0
    while frontier.any():
        d += 1
        nxt = np.zeros_like(frontier)
        nxt[:, 1:, :] |= frontier[:, :-1, :]
        nxt[:, :-1, :] |= frontier[:, 1:, :]
        nxt[:, :, 1:] |= frontier[:, :, :-1]
        nxt[:, :, :-1] |= frontier[:, :, 1:]
        nxt &= free[None] & ~visited
        dist[nxt] = d
        visited |= nxt
        frontier = nxt
    return dist


def build_observations(padded_map, padded_dist, dmax, pos, K, obs_radius, heuristic_norm='relative'):
    pos = np.asarray(pos, dtype=np.int64)
    squeeze = pos.ndim == 2
    if squeeze:
        pos = pos[None]
    T, M, _ = pos.shape
    r = obs_radius
    L = 2 * r + 1
    offset = np.arange(L)
    X = (pos[..., 0, None] + offset)[..., :, None]
    Y = (pos[..., 1, None] + offset)[..., None, :]

    height, width = padded_map.shape
    outside = (X < r) | (X >= height - r) | (Y < r) | (Y >= width - r)
    obstacle = padded_map[X, Y] & ~outside

    t_idx = np.arange(T)
    occupancy = np.zeros((T, *padded_map.shape), dtype=bool)
    occupancy[t_idx[:, None], pos[..., 0] + r, pos[..., 1] + r] = True
    agents = occupancy[t_idx[:, None, None, None], X, Y]
    agents[..., r, r] = False

    diff = pos[:, :, None, :] - pos[:, None, :, :]
    in_fov = np.abs(diff).max(-1) <= r
    eye = np.eye(M, dtype=bool)
    adj = in_fov & ~eye
    sq_dist = (diff ** 2).sum(-1).astype(np.float64)
    sq_dist = np.where(in_fov, sq_dist, np.inf)
    sq_dist[:, eye] = -1
    order = np.argsort(sq_dist, axis=-1, kind='stable')[..., :K]
    nbr = np.where(np.isfinite(np.take_along_axis(sq_dist, order, -1)), order, -1)
    if M < K:
        nbr = np.concatenate([nbr, -np.ones((T, M, K - M), dtype=nbr.dtype)], -1)
    valid = nbr >= 0
    nb = np.maximum(nbr, 0)

    window = padded_dist[nb[..., None, None], X[:, :, None], Y[:, :, None]]
    with np.errstate(invalid='ignore'):
        if heuristic_norm == 'relative':
            center = padded_dist[nb, (pos[..., 0] + r)[..., None], (pos[..., 1] + r)[..., None]]
            heuristic = np.clip((window - center[..., None, None]) / (2 * r), -1, 1)
            bad = ~np.isfinite(window) | ~np.isfinite(center)[..., None, None]
        elif heuristic_norm == 'global':
            heuristic = window / dmax[nb][..., None, None]
            bad = ~np.isfinite(window)
        else:
            raise ValueError('unknown heuristic_norm {}'.format(heuristic_norm))
    heuristic = np.where(bad, 1.0, heuristic)

    obs = np.zeros((T, M, K, NUM_OBS_CHANNELS, L, L), dtype=np.float32)
    obs[:, :, :, 0] = agents[:, :, None]
    obs[:, :, :, 1] = obstacle[:, :, None]
    for d, (dx, dy) in enumerate(DIRECTIONS):
        nx = np.clip(X + dx, 0, height - 1)[:, :, None]
        ny = np.clip(Y + dy, 0, width - 1)[:, :, None]
        obs[:, :, :, 2 + d] = np.isfinite(window) & (padded_dist[nb[..., None, None], nx, ny] < window)
    obs[:, :, :, 6] = heuristic
    obs[:, :, :, 7] = outside[:, :, None]
    obs[~valid] = 0

    if squeeze:
        return obs[0], nbr[0], adj[0]
    return obs, nbr, adj
