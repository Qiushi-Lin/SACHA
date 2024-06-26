import random
import numpy as np
import copy

# config
import yaml
config = yaml.safe_load(open("./config.yaml", 'r'))
FREE_SPACE, OBSTACLE = config['grid_map']['FREE_SPACE'], config['grid_map']['OBSTACLE']
action_mapping = config['action_mapping']


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


def move(loc, d):
    return loc[0] + action_mapping[d][0], loc[1] + action_mapping[d][1]


def map_partition(grid_map):
    empty_spots = np.argwhere(np.array(grid_map)==FREE_SPACE).tolist()
    empty_spots = [tuple(pos) for pos in empty_spots]
    partitions = []
    while empty_spots:
        start_loc = empty_spots.pop()
        open_list = [start_loc]
        close_list = []
        while open_list:
            loc = open_list.pop(0)
            for d in range(4):
                child_loc = move(loc, d)
                if child_loc[0] < 0 or child_loc[0] >= len(grid_map) \
                    or child_loc[1] < 0 or child_loc[1] >= len(grid_map[0]):
                    continue
                if grid_map[child_loc[0]][child_loc[1]] == OBSTACLE:
                    continue
                if child_loc in empty_spots:
                    empty_spots.remove(child_loc)
                    open_list.append(child_loc)
            close_list.append(loc)
        partitions.append(close_list)
    return partitions