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

import time

import chex
import jax
from jax import numpy as jnp

from jumanji.environments.packing.jigsaw.env import Jigsaw
from jumanji.environments.packing.jigsaw.generator import (
    RandomJigsawGenerator,
    ToyJigsawGeneratorNoRotation,
)
from jumanji.environments.packing.jigsaw.reward import SparseReward

SAVE_GIF = False

# Very basic example of a random agent acting in the Jigsaw environment.
# Each episode will generate a completely new instance of the jigsaw puzzle.
env = Jigsaw(
    generator=RandomJigsawGenerator(
        num_col_pieces=5,
        num_row_pieces=5,
    ),
)
action_spec = env.action_spec()
step_key = jax.random.PRNGKey(1)
jit_step = jax.jit(chex.assert_max_traces(env.step, n=1))
jit_reset = jax.jit(chex.assert_max_traces(env.reset, n=1))
episode_returns: list = []
states: list = []
for ep in range(50):
    step_key, reset_key = jax.random.split(step_key)
    state, timestep = jit_reset(key=reset_key)
    states.append(state)
    episode_return = 0
    ep_steps = 0
    start_time = time.time()
    while not timestep.last():
        step_key, piece_key, rot_key, row_key, col_key = jax.random.split(step_key, 5)
        piece_id = jax.random.randint(
            piece_key, shape=(), minval=0, maxval=action_spec.maximum[0] + 1
        )
        rotation = jax.random.randint(
            rot_key, shape=(), minval=0, maxval=action_spec.maximum[1] + 1
        )
        row = jax.random.randint(
            row_key, shape=(), minval=0, maxval=action_spec.maximum[2]
        )
        col = jax.random.randint(
            col_key, shape=(), minval=0, maxval=action_spec.maximum[3]
        )

        action = jnp.array([piece_id, rotation, row, col])
        state, timestep = jit_step(state, action)
        states.append(state)
        episode_return += timestep.reward
        ep_steps += 1

    sps = ep_steps / (time.time() - start_time)
    episode_returns.append(episode_return)
    if ep % 10 == 0:
        print(
            f"EPISODE RETURN: {episode_return}, STEPS PER SECOND: {int(sps)},"
            f" ENVIRONMENT STEPS: {ep_steps}"
        )

print(f"Average return: {jnp.mean(jnp.array(episode_returns))}, SPS: {int(sps)}\n")

if SAVE_GIF:
    env.animate(states=states, interval=200, save_path="big_env.gif")

# An example of solving a puzzle by stepping a
# dummy environment with a dense reward function.
print("STARTING DENSE REWARD EXAMPLE")
env = Jigsaw(generator=ToyJigsawGeneratorNoRotation())
state, timestep = env.reset(step_key)
print("CURRENT BOARD:")
print(state.current_board, "\n")
state, timestep = env.step(state, jnp.array([0, 0, 0, 0]))
print("CURRENT BOARD:")
print(state.current_board, "\n")
print("STEP REWARD:")
print(timestep.reward, "\n")
state, timestep = env.step(state, jnp.array([1, 0, 0, 2]))
print("CURRENT BOARD:")
print(state.current_board, "\n")
print("STEP REWARD:")
print(timestep.reward, "\n")
state, timestep = env.step(state, jnp.array([2, 0, 2, 0]))
print("CURRENT BOARD:")
print(state.current_board, "\n")
print("STEP REWARD:")
print(timestep.reward, "\n")
state, timestep = env.step(state, jnp.array([3, 0, 2, 2]))
print("CURRENT BOARD:")
print(state.current_board, "\n")
print("STEP REWARD:")
print(timestep.reward)
print()

# An example of solving a puzzle by stepping a
# dummy environment with a sparse reward function.
print("STARTING SPARSE REWARD EXAMPLE")
env = Jigsaw(generator=ToyJigsawGeneratorNoRotation(), reward_fn=SparseReward())
state, timestep = env.reset(step_key)
print("CURRENT BOARD:")
print(state.current_board, "\n")
state, timestep = env.step(state, jnp.array([0, 0, 0, 0]))
print("CURRENT BOARD:")
print(state.current_board, "\n")
print("STEP REWARD:")
print(timestep.reward, "\n")
state, timestep = env.step(state, jnp.array([1, 0, 0, 2]))
print("CURRENT BOARD:")
print(state.current_board, "\n")
print("STEP REWARD:")
print(timestep.reward, "\n")
state, timestep = env.step(state, jnp.array([2, 0, 2, 0]))
print("CURRENT BOARD:")
print(state.current_board, "\n")
print("STEP REWARD:")
print(timestep.reward, "\n")
state, timestep = env.step(state, jnp.array([3, 0, 2, 2]))
print("CURRENT BOARD:")
print(state.current_board, "\n")
print("STEP REWARD:")
print(timestep.reward)
