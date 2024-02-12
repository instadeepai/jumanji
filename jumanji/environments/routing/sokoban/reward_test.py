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

import random

import chex
import jax
import jax.numpy as jnp
import pytest

from jumanji.environments.routing.sokoban.env import Sokoban
from jumanji.environments.routing.sokoban.generator import SimpleSolveGenerator
from jumanji.types import TimeStep


@pytest.fixture(scope="session")
def sokoban_simple() -> Sokoban:
    env = Sokoban(generator=SimpleSolveGenerator())
    return env


def test_sokoban__reward_function_random(sokoban_simple: Sokoban) -> None:
    """Check the reward function is correct when randomly acting in the
    trivial problem, where accidently pushing boxes onto targets is likely.
    Every step should give -0.1, each box pushed on adds 1 , each box removed
    on takes away 1 ,solving adds an additional 10"""

    def check_correct_reward(
        timestep: TimeStep,
        num_boxes_on_targets_new: chex.Array,
        num_boxes_on_targets: chex.Array,
    ) -> None:

        if num_boxes_on_targets_new == jnp.array(4, jnp.int32):
            assert timestep.reward == jnp.array(10.9, jnp.float32)
        elif num_boxes_on_targets_new - num_boxes_on_targets > jnp.array(0, jnp.int32):
            assert timestep.reward == jnp.array(0.9, jnp.float32)
        elif num_boxes_on_targets_new - num_boxes_on_targets < jnp.array(0, jnp.int32):
            assert timestep.reward == jnp.array(-1.1, jnp.float32)
        else:
            assert timestep.reward == jnp.array(-0.1, jnp.float32)

    for i in range(5):
        chex.clear_trace_counter()
        step_fn = jax.jit(chex.assert_max_traces(sokoban_simple.step, n=1))

        key = jax.random.PRNGKey(i)
        reset_key, step_key = jax.random.split(key)
        state, timestep = sokoban_simple.reset(reset_key)

        num_boxes_on_targets = sokoban_simple.reward_fn.count_targets(state)

        for _ in range(120):
            action = jnp.array(random.randint(0, 4), jnp.int32)
            state, timestep = step_fn(state, action)

            num_boxes_on_targets_new = sokoban_simple.reward_fn.count_targets(state)

            check_correct_reward(
                timestep, num_boxes_on_targets_new, num_boxes_on_targets
            )

            num_boxes_on_targets = num_boxes_on_targets_new
