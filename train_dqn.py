import os
import copy

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import wandb
wandb.login()
import numpy as np
import torch
import random
np.random.seed(1337)
torch.manual_seed(1337)
random.seed(1337)
from tqdm import tqdm

from src.DQN import DQN
from src.utils import calculate_reward, evaluate_policy, get_greedy_actions
from src.options import INITIAL_STEPS, TRANSITIONS

from world.envs import OnePlayerEnv, VersusBotEnv
from world.map_loaders.single_team import SingleTeamLabyrinthMapLoader, SingleTeamRocksMapLoader
from world.realm import Realm
from world.map_loaders.two_teams import TwoTeamLabyrinthMapLoader, TwoTeamRocksMapLoader
from world.scripted_agents import Dummy, ClosestTargetAgent, BrokenClosestTargetAgent

MIN_REWARD = 0
run = wandb.init(project="rl_preys_predators_2", name="broken_closest_greedy_act_new_reward")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = VersusBotEnv(Realm(TwoTeamRocksMapLoader(), 2, bots={1: BrokenClosestTargetAgent()}))
# env = OnePlayerEnv(Realm(SingleTeamRocksMapLoader(), 1))
dqn = DQN(
    embedding_size=256,
    num_input_channels=6,
    save_path="/home/vk/RL_course_Predators_and_Preys/best_bot_ddqn_closest_bonus_reward_greedy/",
    load_path="/home/vk/RL_course_Predators_and_Preys/best_bot_ddqn_dummy_greedy/",
    device=device,
)
eps_1 = 0.1
eps_2 = 0.3
state, info = env.reset()
dqn.reset(state, info)
processed_state = dqn.preprocess_data(state, info)
for _ in tqdm(range(INITIAL_STEPS)):
    action = np.random.randint(0, 5, size=(5,))
    next_state, done, next_info = env.step(action)
    next_processed_state = dqn.preprocess_data(next_state, next_info)
    reward = calculate_reward(info, next_info, dqn.distance_map, state)
    img, bonuses = processed_state
    new_img, new_bonuses = next_processed_state
    dqn.consume_transition(
        (
            img,
            bonuses,
            action,
            new_img,
            new_bonuses,
            reward,
            done,
        )
    )

    if done:
        state, info = env.reset()
        dqn.reset(state, info)
        processed_state = dqn.preprocess_data(state, info)
    else:
        processed_state = next_processed_state
        state = next_state.copy()
        info = next_info.copy()

state, info = env.reset()
dqn.reset(state, info)
processed_state = dqn.preprocess_data(state, info)

for i in tqdm(range(TRANSITIONS)):
    # Epsilon-greedy policy
    random_number = np.random.rand()
    max_eps_steps = 1e4
    if i < max_eps_steps:
        cur_eps_1 = np.cos(np.pi / 2 * (i + 1) / max_eps_steps) * eps_1
        cur_eps_2 = np.cos(np.pi / 2 * (i + 1) / max_eps_steps) * eps_2
    else:
        cur_eps_1 = 0
        cur_eps_2 = 0

    if random_number < cur_eps_1:
        action = np.random.randint(0, 5, size=(5,))
    elif cur_eps_1 < random_number < cur_eps_2:
        action = get_greedy_actions(state, dqn.distance_map, dqn.action_map)
    else:
        action = dqn.act_preprocessed(processed_state)

    next_state, done, next_info = env.step(action)
    next_processed_state = dqn.preprocess_data(next_state, next_info)

    reward = calculate_reward(info, next_info, dqn.distance_map, state)
    # run.log({"reward": reward})
    img, bonuses = processed_state
    new_img, new_bonuses = next_processed_state
    dqn.update(
        (
            img,
            bonuses,
            action,
            new_img,
            new_bonuses,
            reward,
            done,
        )
    )

    if done:
        state, info = env.reset()
        dqn.reset(state, info)
        processed_state = dqn.preprocess_data(state, info)
    else:
        processed_state = next_processed_state
        state = next_state.copy()
        info = copy.deepcopy(next_info)

    if (i + 1) % (TRANSITIONS // 100) == 0:
        rewards, enemy_rewards = evaluate_policy(dqn, VersusBotEnv(Realm(TwoTeamRocksMapLoader(), 2, bots={1: BrokenClosestTargetAgent()})), episodes=5)
        dqn.reset(state, info)
        our_mean = np.mean(rewards)
        our_std = np.std(rewards)
        enemy_mean = np.mean(enemy_rewards)
        enemy_std = np.std(enemy_rewards)
        print(f"Step: {i + 1}, Our reward mean: {our_mean:.3f}, std: {our_std:.3f}")
        print(f"Step: {i + 1}, Enemy reward mean: {enemy_mean:.3f}, std: {enemy_std:.3f}")
        run.log({
            "Our mean reward": our_mean, 
            "Enemy mean reward": enemy_mean,
            "Reward ratio": our_mean / enemy_mean,
            "Reward difference": our_mean - enemy_mean,
            })
        if our_mean - enemy_mean > MIN_REWARD:
            dqn.save()
            MIN_REWARD = our_mean - enemy_mean

run.finish()
