import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam  # type: ignore
from src.utils import Transition, ReplayMemory
import random

from src.preprocess import RLPreprocessor
from src.options import (
    BATCH_SIZE,
    INITIAL_STEPS,
    LEARNING_RATE,
    GAMMA,
    STEPS_PER_UPDATE,
)
from src.utils import get_bonus_counts

random.seed(1337)


class DQNModel(nn.Module):
    def __init__(self, num_input_channels, embedding_size):
        super().__init__()
        self.data_processor = RLPreprocessor(num_input_channels, embedding_size)
        self.bonus_processor = nn.Linear(1, 32)
        self.layer3 = nn.Linear(embedding_size + 32, 5)

    def forward(self, img, bonuses):
        x = F.relu(self.data_processor(img))
        y = self.bonus_processor(bonuses)
        x = torch.cat((x, y), dim=-1)
        return self.layer3(x)


class DQN:
    def __init__(self, num_input_channels, embedding_size, device: torch.device, save_path="./", load_path=None):
        self.device = device
        self.num_input_channels = num_input_channels
        self.steps = 0  # Do not change

        self.target_model_1 = DQNModel(self.num_input_channels, embedding_size).to(self.device)
        self.policy_model_1 = DQNModel(self.num_input_channels, embedding_size).to(self.device)
        self.target_model_1.load_state_dict(self.policy_model_1.state_dict())

        self.target_model_2 = DQNModel(self.num_input_channels, embedding_size).to(self.device)
        self.policy_model_2 = DQNModel(self.num_input_channels, embedding_size).to(self.device)
        self.target_model_2.load_state_dict(self.policy_model_2.state_dict())

        self.replay_buffer = ReplayMemory(INITIAL_STEPS)
        self.criterion = nn.HuberLoss()
        self.optimizer_1 = Adam(self.policy_model_1.parameters(), lr=LEARNING_RATE)
        self.optimizer_2 = Adam(self.policy_model_2.parameters(), lr=LEARNING_RATE)
        self.tau = 0.005
        self.save_path = save_path
        self.distance_map = None
        self.action_map = None
        if load_path is not None:
            self.load_path = load_path
            self.load()

    def consume_transition(self, transition):
        # Add transition to a replay buffer.
        # Hint: use deque with specified maxlen. It will remove old experience automatically.
        self.replay_buffer.push(transition)

    def sample_batch(self):
        # Sample batch from a replay buffer.
        # Hints:
        # 1. Use random.randint
        # 2. Turn your batch into a numpy.array before turning it to a Tensor. It will work faster
        batch = self.replay_buffer.sample()
        batch = Transition(*zip(*batch))
        return batch

    def train_step(self, batch):
        self.policy_model_1.train()
        # Use batch to update DQN's network.
        img_batch = torch.tensor(np.concatenate(batch.img)).to(torch.float)
        bonuses_batch = torch.tensor(np.concatenate(batch.bonuses)).to(torch.float)
        action_batch = torch.tensor(np.array(batch.action), dtype=torch.int64).to(self.device)

        state_action_values_1 = (
            self.policy_model_1(img_batch.to(self.device), bonuses_batch.to(self.device))
            .gather(dim=2, index=action_batch[..., None])
            .squeeze()
            .to(self.device)
        )

        state_action_values_2 = (
            self.policy_model_2(img_batch.to(self.device), bonuses_batch.to(self.device))
            .gather(dim=2, index=action_batch[..., None])
            .squeeze()
            .to(self.device)
        )

        next_img_batch = torch.tensor(np.concatenate(batch.next_img))
        next_bonuses_batch = torch.tensor(np.concatenate(batch.next_bonuses))
        reward_batch = torch.tensor(np.array(batch.reward)).to(self.device)
        done_batch = torch.tensor(np.array(batch.done))

        non_final_mask = torch.tensor(tuple(map(lambda s: not s, done_batch)), device=self.device, dtype=torch.bool)
        non_final_next_img = (
            next_img_batch.reshape(-1, 5, self.num_input_channels, 40, 40)[~done_batch]
            .reshape(-1, self.num_input_channels, 40, 40)
            .to(torch.float)
            .to(self.device)
        )
        non_final_next_bonuses = next_bonuses_batch[~done_batch].to(torch.float).to(self.device)

        next_state_values_1 = torch.zeros((BATCH_SIZE, 5), device=self.device)
        next_state_values_2 = torch.zeros((BATCH_SIZE, 5), device=self.device)
        with torch.no_grad():
            next_state_values_1[non_final_mask] = (
                self.target_model_1(non_final_next_img, non_final_next_bonuses).max(2).values
            )
            next_state_values_2[non_final_mask] = (
                self.target_model_2(non_final_next_img, non_final_next_bonuses).max(2).values
            )
        
        min_next_state_values = torch.minimum(next_state_values_1, next_state_values_2)

        # Compute the expected Q values
        expected_state_action_values = reward_batch + GAMMA * min_next_state_values

        loss_1 = self.criterion(state_action_values_1.to(torch.float), expected_state_action_values.to(torch.float))
        loss_2 = self.criterion(state_action_values_2.to(torch.float), expected_state_action_values.to(torch.float))
        # print(loss.item())
        # Optimize the model
        self.optimizer_1.zero_grad()
        self.optimizer_2.zero_grad()
        loss_1.backward()
        loss_2.backward()
        torch.nn.utils.clip_grad_value_(self.policy_model_1.parameters(), 50)
        torch.nn.utils.clip_grad_value_(self.policy_model_2.parameters(), 50)
        self.optimizer_1.step()
        self.optimizer_2.step()

    def soft_update_target_network(self):
        target_net_state_dict_1 = self.target_model_1.state_dict()
        policy_net_state_dict_1 = self.policy_model_1.state_dict()
        target_net_state_dict_2 = self.target_model_2.state_dict()
        policy_net_state_dict_2 = self.policy_model_2.state_dict()

        for key_1, key_2 in zip(policy_net_state_dict_1, policy_net_state_dict_2):
            target_net_state_dict_1[key_1] = policy_net_state_dict_1[key_1] * self.tau + target_net_state_dict_1[key_1] * (
                1 - self.tau
            )
            target_net_state_dict_2[key_2] = policy_net_state_dict_2[key_2] * self.tau + target_net_state_dict_2[key_2] * (
                1 - self.tau
            )

        self.target_model_1.load_state_dict(target_net_state_dict_1)
        self.target_model_2.load_state_dict(target_net_state_dict_2)

    def act(self, state, info):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        image, bonuses = self.preprocess_data(state, info)
        cur_state = torch.tensor(image).to(torch.float).to(self.device)
        bonuses = torch.tensor(bonuses).to(self.device)
        self.policy_model_1.eval()
        with torch.no_grad():
            act = self.policy_model_1(cur_state, bonuses).argmax(dim=-1).squeeze().cpu().numpy()
        return act

    def act_preprocessed(self, state):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        image, bonuses = state
        cur_state = torch.tensor(image).to(torch.float).to(self.device)
        bonuses = torch.tensor(bonuses).to(self.device)
        self.policy_model_1.eval()
        with torch.no_grad():
            act = self.policy_model_1(cur_state, bonuses).argmax(dim=-1).squeeze().cpu().numpy()
        return act

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        self.soft_update_target_network()
        self.steps += 1

    def reset(self, initial_state, info):
        mask = np.zeros(initial_state.shape[:2], np.bool_)
        mask[
            np.logical_or(
                np.logical_and(initial_state[:, :, 0] == -1, initial_state[:, :, 1] >= 0), initial_state[:, :, 0] >= 0
            )
        ] = True
        mask = mask.reshape(-1)

        coords_amount = initial_state.shape[0] * initial_state.shape[1]
        self.distance_map = (coords_amount + 1) * np.ones((coords_amount, coords_amount))
        np.fill_diagonal(self.distance_map, 0.0)
        self.distance_map[np.logical_not(mask)] = coords_amount + 1
        self.distance_map[:, np.logical_not(mask)] = coords_amount + 1

        indexes_helper = [
            [
                x * initial_state.shape[1] + (y + 1) % initial_state.shape[1],
                x * initial_state.shape[1] + (initial_state.shape[1] + y - 1) % initial_state.shape[1],
                ((initial_state.shape[0] + x - 1) % initial_state.shape[0]) * initial_state.shape[1] + y,
                ((x + 1) % initial_state.shape[0]) * initial_state.shape[1] + y,
            ]
            for x in range(initial_state.shape[0])
            for y in range(initial_state.shape[1])
        ]

        updated = True
        while updated:
            old_distances = self.distance_map.copy()
            for j in range(coords_amount):
                if mask[j]:
                    for i in indexes_helper[j]:
                        if mask[i]:
                            self.distance_map[j] = np.minimum(self.distance_map[j], self.distance_map[i] + 1)
            updated = (old_distances != self.distance_map).sum() > 0
        
        self.action_map = np.zeros((coords_amount, coords_amount), int)
        for j in range(coords_amount):
            self.action_map[j] = np.argmin(np.stack([self.distance_map[i] + 1 for i in indexes_helper[j]], axis=1),
                                           axis=1) + 1

        self.distance_map = np.where(self.distance_map == (coords_amount + 1), np.nan, self.distance_map)

    def preprocess_data(self, state: np.ndarray, info: dict) -> tuple[np.ndarray, np.ndarray]:
        num_teams = info["preys"][0]["team"]

        hunters_coordinates = np.array([(agent["y"], agent["x"]) for agent in info["predators"]])
        prey_id = state[:, :, 0].max()
        prey_mask = (state[:, :, 0] == prey_id).astype(np.int64)
        hunter_mask = (state[:, :, 0] == 0).astype(np.int64)
        wall_mask = ((state[:, :, 0] == -1) * (state[:, :, 1] == -1)).astype(np.int64)
        bonuses_mask = ((state[:, :, 0] == -1) * (state[:, :, 1] == 1)).astype(np.int64)
        enemy_mask = ((state[:, :, 0] > 0) * (state[:, :, 0] < num_teams)).astype(np.int64)

        for enemy in info["enemy"]:
            enemy_mask[enemy["y"], enemy["x"]] *= enemy["bonus_count"] + 1

        states = []
        for hunter_coordinates in hunters_coordinates:
            centred_coods = 20 - hunter_coordinates[0], 20 - hunter_coordinates[1]
            obst_mask = np.roll(wall_mask, centred_coods, axis=(0, 1))
            distance_mask = np.roll(
                self.distance_map[hunter_coordinates[0] * 40 + hunter_coordinates[1]].reshape(40, 40),  # type: ignore
                centred_coods,
                axis=(0, 1),
            )
            distance_mask = np.nan_to_num(distance_mask, nan=-1)
            distance_mask = distance_mask / distance_mask.max()
            distance_mask = np.where(distance_mask < 0, 2, distance_mask)
            hunter_state = np.stack(
                (
                    np.roll(prey_mask, centred_coods, axis=(0, 1)),
                    np.roll(hunter_mask, centred_coods, axis=(0, 1)),
                    obst_mask,
                    distance_mask,
                    np.roll(bonuses_mask, centred_coods, axis=(0, 1)),
                    np.roll(enemy_mask, centred_coods, axis=(0, 1)),
                ),
            )
            states.append(hunter_state.astype(np.float64))

        bonus_counts = get_bonus_counts(info)[None, :, None].astype(np.float32)

        return np.stack(states), bonus_counts

    def save(self):
        if self.save_path:
            torch.save(self.policy_model_1.state_dict(), self.save_path + "agent.pkl")

    def load(self):
        self.policy_model_1.load_state_dict(torch.load(self.load_path + "agent.pkl", map_location=self.device))
        self.target_model_1.load_state_dict(torch.load(self.load_path + "agent.pkl", map_location=self.device))
