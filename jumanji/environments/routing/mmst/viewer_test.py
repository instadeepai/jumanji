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

import os
from typing import Tuple

import jax
import jax.numpy as jnp

from jumanji.environments.routing.mmst.env import MMST
from jumanji.environments.routing.mmst.types import State
from jumanji.environments.routing.mmst.viewer import MMSTViewer
from jumanji.types import TimeStep


def test__render(deterministic_mmst_env: Tuple[MMST, State, TimeStep]) -> None:
    """Test that viewer works and the frame is saved."""

    env, state, _ = deterministic_mmst_env
    viewer = MMSTViewer(
        env.num_agents,
    )

    viewer.render(state)


def test__animation(deterministic_mmst_env: Tuple[MMST, State, TimeStep]) -> None:
    """Test the viewer's animation function."""

    env, state, _ = deterministic_mmst_env
    viewer = MMSTViewer(
        env.num_agents,
    )
    step_fn = jax.jit(env.step)
    save_path = "render_test_gif.gif"

    states = [state]
    actions = [
        jnp.array([0, 4]),
        jnp.array([3, 5]),
        jnp.array([7, 4]),
        jnp.array([6, 8]),
        jnp.array([10, 10]),
        jnp.array([9, 4]),
    ]

    for action in actions:
        state, _ = step_fn(state, action)
        states.append(state)

    viewer.animate(states, save_path=save_path)

    assert os.path.exists(save_path)
    os.remove(save_path)
