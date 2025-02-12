# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import jax
import requests
from hydra import compose, initialize

from jumanji.training.setup_train import setup_agent, setup_env

envs = [
    "bin_pack",
    "cleaner",
    "connector",
    "cvrp",
    "flat_pack",
    "game_2048",
    "graph_coloring",
    "job_shop",
    "knapsack",
    "maze",
    "minesweeper",
    "mmst",
    "multi_cvrp",
    "pac_man",
    "robot_warehouse",
    "lbf",
    "rubiks_cube",
    "sliding_tile_puzzle",
    "snake",
    "sokoban",
    "sudoku",
    "tetris",
    "tsp",
]


def download_file(url: str, file_path: str) -> None:
    # Send an HTTP GET request to the URL
    response = requests.get(url)
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
    else:
        print("Failed to download the file.")


def create_animation(env_name: str, agent: str = "random", num_episodes: int = 2) -> None:
    print(f"Animating {env_name}")

    os.makedirs("configs", exist_ok=True)
    config_url = "https://raw.githubusercontent.com/instadeepai/jumanji/main/jumanji/training/configs/config.yaml"
    download_file(config_url, "configs/config.yaml")
    env_url = f"https://raw.githubusercontent.com/instadeepai/jumanji/main/jumanji/training/configs/env/{env_name}.yaml"
    os.makedirs("configs/env", exist_ok=True)
    download_file(env_url, f"configs/env/{env_name}.yaml")
    os.makedirs("animations", exist_ok=True)

    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="config.yaml", overrides=[f"env={env_name}", f"agent={agent}"])

    env = setup_env(cfg).unwrapped
    agent = setup_agent(cfg, env)
    policy = jax.jit(agent.make_policy(stochastic=False))

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    states = []
    key = jax.random.PRNGKey(cfg.seed)

    for _ in range(num_episodes):
        key, reset_key = jax.random.split(key)
        state, timestep = reset_fn(reset_key)
        states.append(state)

        while not timestep.last():
            key, action_key = jax.random.split(key)
            observation = jax.tree_util.tree_map(lambda x: x[None], timestep.observation)
            action = policy(observation, action_key)
            state, timestep = step_fn(state, action.squeeze(axis=0))
            states.append(state)

    env.animate(states, 100, f"animations/{env_name}_animation.gif")


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument(
        "envs",
        nargs="*",
        type=str,
        default=None,
    )

    args = cli.parse_args()
    arg_envs = args.envs
    env_list = envs if len(arg_envs) == 0 else args.envs

    for env in env_list:
        try:
            create_animation(env)
        except Exception as e:
            print(f"{env} failed", e)
