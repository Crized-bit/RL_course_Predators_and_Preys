import math
from typing import Optional, Union, Sequence

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.functional import center_crop


class NoisyLinear(nn.Linear):
    """Noisy Linear Layer.

    Presented in "Noisy Networks for Exploration", https://arxiv.org/abs/1706.10295v3

    A Noisy Linear Layer is a linear layer with parametric noise added to the weights. This induced stochasticity can
    be used in RL networks for the agent's policy to aid efficient exploration. The parameters of the noise are learned
    with gradient descent along with any other remaining network weights. Factorized Gaussian
    noise is the type of noise usually employed.


    Args:
        in_features (int): input features dimension
        out_features (int): out features dimension
        bias (bool, optional): if ``True``, a bias term will be added to the matrix multiplication: Ax + b.
            Defaults to ``True``
        device (DEVICE_TYPING, optional): device of the layer.
            Defaults to ``"cpu"``
        dtype (torch.dtype, optional): dtype of the parameters.
            Defaults to ``None`` (default pytorch dtype)
        std_init (scalar, optional): initial value of the Gaussian standard deviation before optimization.
            Defaults to ``0.1``

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional = None,
        dtype: Optional[torch.dtype] = None,
        std_init: float = 0.1,
    ):
        nn.Module.__init__(self)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.std_init = std_init

        self.weight_mu = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        self.weight_sigma = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        self.register_buffer(
            "weight_epsilon",
            torch.empty(out_features, in_features, device=device, dtype=dtype),
        )
        if bias:
            self.bias_mu = nn.Parameter(
                torch.empty(
                    out_features,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
            self.bias_sigma = nn.Parameter(
                torch.empty(
                    out_features,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
            self.register_buffer(
                "bias_epsilon",
                torch.empty(out_features, device=device, dtype=dtype),
            )
        else:
            self.bias_mu = None
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        if self.bias_mu is not None:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self) -> None:
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        if self.bias_mu is not None:
            self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: Union[int, torch.Size, Sequence]) -> torch.Tensor:
        if isinstance(size, int):
            size = (size,)
        x = torch.randn(*size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    @property
    def weight(self) -> torch.Tensor:
        if self.training:
            return self.weight_mu + self.weight_sigma * self.weight_epsilon
        else:
            return self.weight_mu

    @property
    def bias(self) -> Optional[torch.Tensor]:
        if self.bias_mu is not None:
            if self.training:
                return self.bias_mu + self.bias_sigma * self.bias_epsilon
            else:
                return self.bias_mu
        else:
            return None


def get_bonus_counts(info):
    return np.array([p["bonus_count"] for p in info["predators"]])


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act = nn.LeakyReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ResConvBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.LeakyReLU()
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.act(y)
        return x + y


class ImagePreprocessor(nn.Module):
    def __init__(self, num_input_channels, embedding_size):
        super(ImagePreprocessor, self).__init__()
        # 40 40 2
        self.full_conv = nn.Sequential(
            ConvBlock(in_channels=num_input_channels, out_channels=8, kernel_size=3, stride=1, padding=1),
            ResConvBlock(8),
            ResConvBlock(8),
            ResConvBlock(8),
            ConvBlock(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            ResConvBlock(16),
            ResConvBlock(16),
            ResConvBlock(16),
            ConvBlock(in_channels=16, out_channels=num_input_channels, kernel_size=3, stride=1, padding=1),
            ResConvBlock(num_input_channels),
            ResConvBlock(num_input_channels),
            nn.Flatten(),
            NoisyLinear(40 * 40 * num_input_channels, 256),
        )

        self.size_10_conv = nn.Sequential(
            ResConvBlock(num_input_channels),
            ResConvBlock(num_input_channels),
            ResConvBlock(num_input_channels),
        )
        self.size_10_lin = NoisyLinear(10 * 10 * num_input_channels, 256)

        self.size_5_conv = nn.Sequential(
            ResConvBlock(num_input_channels),
            ResConvBlock(num_input_channels),
        )
        self.size_5_lin = NoisyLinear(5 * 5 * num_input_channels, 256)

        self.linear_last = NoisyLinear(256 * 3, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_5 = self.size_5_lin(self.size_5_conv(center_crop(x, [5, 5])).flatten(start_dim=1))
        x_10 = self.size_10_lin(self.size_10_conv(center_crop(x, [10, 10])).flatten(start_dim=1))
        x_full = self.full_conv(x)
        result = torch.cat((x_full, x_10, x_5), dim=-1)
        return self.linear_last(result)


class RLPreprocessor(nn.Module):
    def __init__(self, num_input_channels, embedding_size):
        super(RLPreprocessor, self).__init__()
        self.ImagePreprocessor = ImagePreprocessor(num_input_channels, embedding_size)
        # self.linear = nn.Linear(256 + 2, 256)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # addtional_features.shape [N_batch, 5, 2]
        # image_features.shape [N_batch, 256] -> [N_batch, 5, 256]
        # N_batch, n_agents, state_dim
        image_features = self.ImagePreprocessor(image)
        return image_features.reshape(-1, 5, image_features.shape[-1])


class DQNModel(nn.Module):
    def __init__(self, num_input_channels, embedding_size):
        super().__init__()
        self.data_processor = RLPreprocessor(num_input_channels, embedding_size)
        self.bonus_processor = NoisyLinear(1, 32)
        self.layer3 = NoisyLinear(embedding_size + 32, 5)

    def forward(self, img, bonuses):
        x = F.relu(self.data_processor(img))
        y = self.bonus_processor(bonuses)
        x = torch.cat((x, y), dim=-1)
        return self.layer3(x)


class Agent:
    def __init__(self) -> None:
        self.agent = DQNModel(6, 256)
        self.agent.load_state_dict(torch.load(__file__[:-8] + "/agent.pkl", map_location="cpu"))
        self.distance_map = None

    def get_actions(self, state, info):
        return list(self.act(state, info))

    def act(self, state, info):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        image, bonuses = self.preprocess_data(state, info)
        cur_state = torch.tensor(image).to(torch.float)
        bonuses = torch.tensor(bonuses)
        self.agent.eval()
        with torch.no_grad():
            act = self.agent(cur_state, bonuses).argmax(dim=-1).squeeze().cpu().numpy()
        return act

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
