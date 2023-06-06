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
import jax.numpy as jnp
import numpy as np

from jumanji.environments.routing.multi_cvrp.env import MultiCVRP, Observation
from jumanji.environments.routing.multi_cvrp.viewer import MultiCVRPViewer


def test_render(multicvrp_env: MultiCVRP) -> None:
    """Test that viewer works and the frame is saved."""

    key = jax.random.PRNGKey(0)
    reset_fn = jax.jit(multicvrp_env.reset)
    step_fn = jax.jit(multicvrp_env.step)
    state, timestep = reset_fn(key)

    viewer = MultiCVRPViewer(
        name="MultiCVRP",
        num_vehicles=multicvrp_env._num_vehicles,
        num_customers=multicvrp_env._num_customers,
        map_max=multicvrp_env._map_max,
        render_mode="human",
    )

    # Starting position is depot, new action to visit first node
    new_actions = jnp.array(
        jnp.arange(1, multicvrp_env._num_vehicles + 1), dtype=np.int16
    )

    new_state, next_timestep = step_fn(state, new_actions)

    save_path = "render_test.png"
    viewer.render(new_state, save_path=save_path)

    assert os.path.exists(save_path)
    os.remove(save_path)


def test_animation(multicvrp_env: MultiCVRP) -> None:
    """Test the viewer's animation function."""

    def select_actions(key: chex.PRNGKey, observation: Observation) -> chex.Array:
        @jax.vmap  # map over the agents
        def select_action(
            key: chex.PRNGKey, agent_action_mask: chex.Array
        ) -> chex.Array:
            return jnp.array(
                jax.random.choice(
                    key,
                    jnp.arange(len(agent_action_mask)),
                    p=agent_action_mask.flatten(),
                ),
                dtype=np.int16,
            )

        subkeys = jax.random.split(key, multicvrp_env._num_vehicles)
        return select_action(subkeys, observation.action_mask)

    key = jax.random.PRNGKey(1)
    reset_fn = jax.jit(multicvrp_env.reset)
    step_fn = jax.jit(multicvrp_env.step)
    state, timestep = reset_fn(key)
    viewer = MultiCVRPViewer(
        name="MultiCVRP",
        num_vehicles=multicvrp_env._num_vehicles,
        num_customers=multicvrp_env._num_customers,
        map_max=multicvrp_env._map_max,
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
