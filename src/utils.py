import numpy as np
import random
from collections import namedtuple, deque
from numba import jit

from src.options import BATCH_SIZE
from world.utils import RenderedEnvWrapper

def calculate_reward(old_info, new_info, distance_map, old_state) -> np.ndarray:
    # eaten словарь из пойманных существ.
    # Ключи - номер команды и индекс пойманного существа,
    # значение - номер команды и индекс существа, которое его поймало

    old_hunter_coords = np.array([(agent["y"], agent["x"]) for agent in old_info["predators"]])
    new_hunter_coords = np.array([(agent["y"], agent["x"]) for agent in new_info["predators"]])

    enemy_coords = np.array([(enemy["y"], enemy["x"]) for enemy in old_info["enemy"] if enemy["alive"]])
    pray_coords = np.array([(prey["y"], prey["x"]) for prey in old_info["preys"] if prey["alive"]])

    eatable_targets = np.concatenate((pray_coords, enemy_coords))

    old_distances = get_targets_distance(old_hunter_coords, eatable_targets, distance_map)
    new_distances = get_targets_distance(new_hunter_coords, eatable_targets, distance_map)

    values = np.ones(len(eatable_targets))
    values[-len(enemy_coords) :] = values[-len(enemy_coords) :] * 3  # As hunter kill equals 3 points +-

    density = get_targets_density(eatable_targets, distance_map, values, radius=5)

    enemy_bonus_counts = np.array([enemy["bonus_count"] for enemy in old_info["enemy"] if enemy["alive"]])
    agents_bonus_counts = get_bonus_counts(old_info)

    # Because we can't kill instantly and need 1 + bonus turns to achieve
    density[-len(enemy_coords):] = density[-len(enemy_coords):] * (1 / (enemy_bonus_counts + 1))

    metric = density[None, :] / (old_distances + 1)  # [5, n_pray] and fixes div 0 error

    # We should not eat enemy when we have no bonus
    metric[:, -len(enemy_coords) :] = metric[:, -len(enemy_coords) :] * (agents_bonus_counts[:, None] > enemy_bonus_counts[None, :])

    best_pray_index = np.argmax(metric, axis=1)

    old_distances = np.take_along_axis(old_distances, best_pray_index[:, None], axis=1).squeeze()
    new_distances = np.take_along_axis(new_distances, best_pray_index[:, None], axis=1).squeeze()

    isnan = np.logical_or(np.isnan(old_distances), np.isnan(new_distances))
    old_distances[isnan] = 0
    new_distances[isnan] = 0

    dist_difference = new_distances - old_distances

    prey_kills, enemy_kills, bonus_kills = get_kills(old_info, new_info)

    killed_anybody = np.logical_or(prey_kills, enemy_kills, bonus_kills)

    dist_difference[killed_anybody == 1] = 0

    sudden_change = np.logical_or(dist_difference > 2, dist_difference < -2)
    dist_difference[sudden_change] = 0

    dist_difference = np.clip(dist_difference, -1, 1)
    
    reward_for_bonus = get_bonus_reward(old_state, old_hunter_coords, new_hunter_coords, distance_map)
    reward_for_bonus[(killed_anybody == 1) & (reward_for_bonus < 0)] = 0

    reward_for_distances = get_distance_reward(new_hunter_coords, distance_map)
    
    result = (
        dist_difference * -0.5 +
        prey_kills * 2.0 +
        bonus_kills * 1.7 +
        enemy_kills * (agents_bonus_counts > 0) * 1.5 +
        reward_for_bonus * 0.25 +
        reward_for_distances * 0.25
    )

    stands_still = check_for_standing_still(old_info, new_info)
    result[stands_still == 1] = -0.7

    return result


def get_bonus_reward(old_state, old_hunter_coords, new_hunter_coords, distance_map):
    bonus_mask = ((old_state[:, :, 0] == -1) * (old_state[:, :, 1] == 1)).astype(np.int64)
    y_bonus_coords, x_bonus_coords = np.nonzero(bonus_mask)
    old_predators_distances = distance_map[old_hunter_coords[:, 0] * 40 + old_hunter_coords[:, 1]]
    new_predators_distances = distance_map[new_hunter_coords[:, 0] * 40 + new_hunter_coords[:, 1]]
    old_predators2bonuses_distances = np.take_along_axis(old_predators_distances, (y_bonus_coords * 40 + x_bonus_coords)[None, :], axis=1).squeeze()
    old_predators2bonuses_distances = np.nan_to_num(old_predators2bonuses_distances, nan=float("inf"))
    old_bonus_numbers = np.argmin(old_predators2bonuses_distances, axis=1)

    new_predators2bonuses_distances = np.take_along_axis(new_predators_distances, (y_bonus_coords * 40 + x_bonus_coords)[None, :], axis=1).squeeze()

    return np.min(old_predators2bonuses_distances, axis=1) - np.take_along_axis(new_predators2bonuses_distances, old_bonus_numbers[:, None], axis=1).squeeze()


def get_distance_reward(new_hunter_coords, distance_map):
    new_predators_distances = distance_map[new_hunter_coords[:, 0] * 40 + new_hunter_coords[:, 1]]
    distances_between_hunters = np.take_along_axis(new_predators_distances, (new_hunter_coords[:, 0] * 40 + new_hunter_coords[:, 1])[None, :], axis=1).squeeze()
    np.fill_diagonal(distances_between_hunters, float("inf"))
    reward = distances_between_hunters.min(-1) / 20

    return reward


def check_for_standing_still(info, next_info):
    out = []
    for predator_info, next_predator_info in zip(info["predators"], next_info["predators"]):
        out.append(predator_info["x"] == next_predator_info["x"] and predator_info["y"] == next_predator_info["y"])
    return np.array(out)


def get_bonus_counts(info):
    return np.array([p["bonus_count"] for p in info["predators"]])


def get_kills(info, next_info):
    """Returns prey kills and enemy kills for each predator of team 0 during the step"""
    n = len(next_info["predators"])
    prey_team_id = next_info["preys"][0]["team"]
    prey_kills = np.zeros(n)
    enemy_kills = np.zeros(n)

    for killed, killer in next_info["eaten"].items():
        if killer[0] != 0:
            continue

        if killed[0] == prey_team_id:
            prey_kills[killer[1]] = 1
        else:
            enemy_kills[killer[1]] = 1

    bonus_counts = get_bonus_counts(info)
    bonus_counts_next = get_bonus_counts(next_info)
    bonus_kills = (bonus_counts_next > bonus_counts).astype(int)

    return prey_kills, enemy_kills, bonus_kills


@jit(nopython=True, parallel=True)
def get_targets_distance(hunters_coordinates: np.ndarray, preys_coords: np.ndarray, distance_map: np.ndarray):
    targets_distance = np.empty(shape=(5, len(preys_coords)), dtype=np.int64)
    for i, hunter_coordinates in enumerate(hunters_coordinates):
        distances = distance_map[hunter_coordinates[0] * 40 + hunter_coordinates[1], :]
        flat_indices = np.array([prey_coords[0] * 40 + prey_coords[1] for prey_coords in preys_coords])
        distances = distances[flat_indices]
        targets_distance[i] = distances

    return targets_distance


@jit(nopython=True, parallel=True)
def get_targets_density(eatable_targets_coords: np.ndarray, distance_map: np.ndarray, values: np.ndarray, radius):
    density = []
    flat_indices = np.array([target_coords[0] * 40 + target_coords[1] for target_coords in eatable_targets_coords])
    for target_coords in eatable_targets_coords:
        distances = distance_map[target_coords[0] * 40 + target_coords[1]]
        distances = distances[flat_indices]
        density.append(np.sum(values[distances < radius]))
    return np.array(density)


Transition = namedtuple(
    typename="Transition", field_names=("img", "bonuses", "action", "next_img", "next_bonuses", "reward", "done")
)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args[0]))

    def sample(self, batch_size=BATCH_SIZE):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def evaluate_policy(agent, env, do_render=False, episodes=5):
    if do_render:
        env = RenderedEnvWrapper(env)
    scores = []
    enemy_scores = []
    for i in range(episodes):
        done = False
        state, info = env.reset()
        agent.reset(state, info)

        while not done:
            state, done, info = env.step(agent.act(state, info))
        if do_render:
            env.render(f"./Episode_{i+1}")
        scores.append(info["scores"][0])
        if len(info["scores"]) > 1:
            enemy_scores.append(info["scores"][1])
        else:
            enemy_scores.append(0)
    return scores, enemy_scores


@jit(nopython=True, parallel=True)
def get_greedy_actions(state, distance_map, action_map):
    actions = np.zeros(5, dtype=np.int32)
    predators = np.zeros((5, 2), dtype=np.int32)
    preys = []
    preys_team = 2
    if preys_team == 0:
        preys_team = None
    for x in range(state.shape[0]):
        for y in range(state.shape[1]):
            if state[x, y, 0] == 0:
                predators[state[x, y, 1]] = np.array([x, y])
                continue
            if (preys_team is None and state[x, y, 0] > 0) or (state[x, y, 0] == preys_team):
                preys.append(np.array([x, y]))
    if len(preys) > 0:
        for k, (xs, ys) in enumerate(predators):
            target = preys[0]
            for p in preys:
                if (distance_map[xs * state.shape[1] + ys, p[0] * state.shape[1] + p[1]] <
                        distance_map[xs * state.shape[1] + ys, target[0] * state.shape[1] + target[1]]):
                    target = p
            actions[k] = action_map[xs * state.shape[1] + ys, target[0] * state.shape[1] + target[1]]
    return actions