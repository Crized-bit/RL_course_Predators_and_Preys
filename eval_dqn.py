import numpy as np
import torch
import imageio

from pathlib import Path
from src.DQN import DQN
from src.utils import evaluate_policy

from world.envs import OnePlayerEnv, VersusBotEnv
from world.map_loaders.single_team import SingleTeamLabyrinthMapLoader, SingleTeamRocksMapLoader
from world.realm import Realm
from world.map_loaders.two_teams import TwoTeamLabyrinthMapLoader, TwoTeamRocksMapLoader
from world.scripted_agents import Dummy, ClosestTargetAgent

np.random.seed(1337)
torch.manual_seed(1337)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
env = VersusBotEnv(Realm(TwoTeamRocksMapLoader(preys_num=10), 2, bots={1: Dummy()}))
dqn = DQN(
    embedding_size=256,
    num_input_channels=6,
    #   save_path="/home/RL_course_Predators_and_Preys/best_bot_vs_normal/",
    load_path="/home/vk/RL_course_Predators_and_Preys/best_bot_ddqn_dummy_greedy/",
    device=device,
)

rewards, enemy_rewards = evaluate_policy(dqn, env, episodes=1, do_render=True)

with imageio.get_writer("/home/vk/Episode_1/0_movie.gif", mode="I", duration=0.5) as writer:
    for filename in sorted(list(Path("/home/vk/Episode_1/").glob("*.png")), key=lambda x: int(x.stem)):
        print(filename)
        image = imageio.imread(filename)
        writer.append_data(image)  # type: ignore

print(f"Reward mean: {np.mean(rewards)}, std: {np.std(rewards)}")
print(f"Enemy reward mean: {np.mean(enemy_rewards)}, std: {np.std(enemy_rewards)}")
