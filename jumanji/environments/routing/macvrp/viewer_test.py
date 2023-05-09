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

import chex
import jax
import numpy as np

from jumanji.environments.routing.macvrp.env import MACVRP, Observation
from jumanji.environments.routing.macvrp.viewer import MACVRPViewer


def test_render(macvrp_env: MACVRP) -> None:
    """Test that viewer works and the frame is saved."""

    key = jax.random.PRNGKey(0)
    reset_fn = jax.jit(macvrp_env.reset)
    step_fn = jax.jit(macvrp_env.step)
    state, timestep = reset_fn(key)

    viewer = MACVRPViewer(
        name="MACVRP",
        num_vehicles=macvrp_env.num_vehicles,
        num_customers=macvrp_env.num_customers,
        map_max=macvrp_env.map_max,
        render_mode="human",
    )

    # Starting position is depot, new action to visit first node
    new_actions = jax.numpy.array(
        jax.numpy.arange(1, macvrp_env.num_vehicles + 1), dtype=np.int16
    )

    new_state, next_timestep = step_fn(state, new_actions)

    save_path = "render_test.png"
    viewer.render(new_state, save_path=save_path)

    assert os.path.exists(save_path)
    os.remove(save_path)


def test_animation(macvrp_env: MACVRP) -> None:
    """Test the viewer's animation function."""

    def select_actions(key: chex.PRNGKey, observation: Observation) -> chex.Array:
        @jax.vmap  # map over the agents
        def select_action(
            key: chex.PRNGKey, agent_action_mask: chex.Array
        ) -> chex.Array:
            return jax.numpy.array(
                jax.random.choice(
                    key,
                    jax.numpy.arange(len(agent_action_mask)),
                    p=agent_action_mask.flatten(),
                ),
                dtype=np.int16,
            )

        subkeys = jax.random.split(key, macvrp_env.num_vehicles)
        return select_action(subkeys, observation.action_mask)

    key = jax.random.PRNGKey(1)
    reset_fn = jax.jit(macvrp_env.reset)
    step_fn = jax.jit(macvrp_env.step)
    state, timestep = reset_fn(key)
    viewer = MACVRPViewer(
        name="MACVRP",
        num_vehicles=macvrp_env.num_vehicles,
        num_customers=macvrp_env.num_customers,
        map_max=macvrp_env.map_max,
        render_mode="human",
    )

    save_path = "render_test.gif"
    states = [state]

    for _ in range(10):
        key, use_key = jax.random.split(key)
        actions = select_actions(use_key, timestep.observation)
        state, timestep = step_fn(state, actions)
        states.append(state)

    viewer.animate(states, save_path=save_path)

    assert os.path.exists(save_path)
    os.remove(save_path)
