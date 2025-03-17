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

import chex
import jax
import numpy as np

from jumanji.environments.routing.multi_cvrp.constants import DEPOT_IDX
from jumanji.environments.routing.multi_cvrp.env import MultiCVRP, Observation
from jumanji.environments.routing.multi_cvrp.types import State
from jumanji.testing.env_not_smoke import (
    check_env_does_not_smoke,
    check_env_specs_does_not_smoke,
)
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.types import TimeStep


class TestEnvironmentSpec:
    def test_multicvrp__reset(self, multicvrp_env: MultiCVRP) -> None:
        """Validates the jitted reset of the environment."""
        chex.clear_trace_counter()
        reset_fn = jax.jit(chex.assert_max_traces(multicvrp_env.reset, n=1))

        key = jax.random.PRNGKey(0)
        state, timestep = reset_fn(key)

        # # Call again to check it does not compile twice.
        state, timestep = reset_fn(key)

        assert isinstance(timestep, TimeStep)
        assert isinstance(state, State)

        # Initial position is at depot, so current capacity is max capacity
        assert jax.numpy.all(state.vehicles.capacities == multicvrp_env._max_capacity)

        # # All vechicles are at the depot
        assert jax.numpy.all(state.vehicles.positions == DEPOT_IDX)

        # Cost of the depot must be 0
        assert state.nodes.demands[DEPOT_IDX] == 0

        # Check that the time windows are valid.
        assert jax.numpy.all(0 < state.windows.start)
        assert jax.numpy.all(state.windows.start < multicvrp_env._max_start_window)
        assert jax.numpy.all(state.windows.end > state.windows.start)

        # Check that the early_coefs and late_coefs are valid.
        assert jax.numpy.all(multicvrp_env._early_coef_rand[0] <= state.coeffs.early)
        assert jax.numpy.all(state.coeffs.early < multicvrp_env._early_coef_rand[1])
        assert jax.numpy.all(multicvrp_env._late_coef_rand[0] <= state.coeffs.late)
        assert jax.numpy.all(state.coeffs.late < multicvrp_env._late_coef_rand[1])

        assert_is_jax_array_tree(state)

    def test_multicvrp__step(self, multicvrp_env: MultiCVRP) -> None:
        """Validates the jitted step of the environment."""
        chex.clear_trace_counter()

        key = jax.random.PRNGKey(0)
        reset_fn = jax.jit(chex.assert_max_traces(multicvrp_env.reset, n=1))
        state, timestep = reset_fn(key)

        # Check the timestep step type is FIRST
        assert timestep.first()
        # Starting position is depot, new action to visit first node
        new_actions = jax.numpy.array(
            jax.numpy.arange(1, multicvrp_env._num_vehicles + 1), dtype=np.int16
        )

        step_fn = jax.jit(chex.assert_max_traces(multicvrp_env.step, n=1))
        new_state, next_timestep = step_fn(state, new_actions)

        # Check that the state has changed
        assert not jax.numpy.array_equal(
            new_state.vehicles.distances, state.vehicles.distances
        )  # Some time penalties are 0

        for i in range(multicvrp_env._num_vehicles):
            assert not jax.numpy.array_equal(
                new_state.vehicles.positions[i], state.vehicles.positions[i]
            )
            assert not jax.numpy.array_equal(
                new_state.vehicles.capacities[i], state.vehicles.capacities[i]
            )
            assert not jax.numpy.array_equal(
                new_state.vehicles.local_times[i], state.vehicles.local_times[i]
            )
            assert not jax.numpy.array_equal(
                new_state.vehicles.distances[i], state.vehicles.distances[i]
            )

        # Check that the state is made of DeviceArrays, this is false for the non-jitted
        # step function since unpacking random.split returns numpy arrays and not device arrays.
        assert_is_jax_array_tree(new_state)

        # Take another set of valid actions.
        new_actions: list = []
        node_i = multicvrp_env._num_vehicles
        while len(new_actions) < multicvrp_env._num_vehicles:
            if new_state.nodes.demands[node_i] > 0:
                new_actions.append(node_i)
            node_i += 1
            if node_i >= multicvrp_env._num_customers:
                raise ValueError("There is not enough customer demand for a second action..")
        new_actions = jax.numpy.array(new_actions, dtype=np.int16)

        # # Take the same actions again which should now be invalid.
        state = new_state

        new_state, next_timestep = step_fn(state, new_actions)

        # Check that the timestep is mid.
        assert next_timestep.mid()

        # New step with same action should be invalid.
        state = new_state

        new_state, next_timestep = step_fn(state, new_actions)

        # Check that all the agents are at the depot.
        assert np.all(new_state.vehicles.positions == 0)

    def test_multicvrp__update_state(self, multicvrp_env: MultiCVRP) -> None:
        """Validates the jitted step of the environment."""
        chex.clear_trace_counter()

        _update_state_fn = jax.jit(chex.assert_max_traces(multicvrp_env._update_state, n=1))

        key = jax.random.PRNGKey(0)
        state, _ = multicvrp_env.reset(key)

        # Check the order and step_count
        assert np.all(state.order == 0)
        assert np.all(state.step_count == 1)

        # Check that the agents are all at the depot
        assert jax.numpy.array_equal(
            state.vehicles.positions, jax.numpy.array([0, 0], dtype=jax.numpy.int16)
        )

        # Starting position is depot, new action to visit first node
        new_actions = jax.numpy.array(
            jax.numpy.arange(1, multicvrp_env._num_vehicles + 1), dtype=np.int16
        )

        new_state = _update_state_fn(state, new_actions)

        # Check the order and step_count
        old_actions = new_actions
        assert np.array_equal(new_state.order[:, 1], new_actions)
        assert np.all(new_state.step_count == 2)

        # Check that the agents moved
        assert jax.numpy.array_equal(
            new_state.vehicles.positions, jax.numpy.array([1, 2], dtype=jax.numpy.int16)
        )

        # Check that the node coordinates remained the same
        assert jax.numpy.array_equal(state.nodes.coordinates, new_state.nodes.coordinates)

        new_actions = jax.numpy.array([0, 0], dtype=np.int16)

        new_state = _update_state_fn(new_state, new_actions)

        # Check the order and step_count
        assert np.array_equal(new_state.order[:, 1], old_actions)
        assert np.array_equal(new_state.order[:, 2], new_actions)
        assert np.all(new_state.order[:, 2:] == 0)
        assert np.all(new_state.step_count == 3)

        # Check that the agents are all at the depot again
        assert jax.numpy.array_equal(
            state.vehicles.positions, jax.numpy.array([0, 0], dtype=jax.numpy.int16)
        )

    def test_multicvrp__state_to_observation_timestep(self, multicvrp_env: MultiCVRP) -> None:
        """Validates the jitted step of the environment."""
        chex.clear_trace_counter()

        update_state_fn = jax.jit(chex.assert_max_traces(multicvrp_env._update_state, n=1))
        state_to_observation_fn = jax.jit(
            chex.assert_max_traces(multicvrp_env._state_to_observation, n=1)
        )
        state_to_timestep_fn = jax.jit(
            chex.assert_max_traces(multicvrp_env._state_to_timestep, n=1)
        )

        reward_fn = jax.jit(chex.assert_max_traces(multicvrp_env._reward_fn.__call__, n=1))

        key = jax.random.PRNGKey(0)
        state, _ = multicvrp_env.reset(key)

        # Starting position is depot, new action to visit first node
        new_actions = jax.numpy.array(
            jax.numpy.arange(1, multicvrp_env._num_vehicles + 1), dtype=np.int16
        )

        new_state = update_state_fn(state, new_actions)

        obs = state_to_observation_fn(new_state)

        # Check that the node coordinates are duplicated correctly
        assert np.array_equal(
            obs.nodes.coordinates[0], state.nodes.coordinates[0]
        ) and np.array_equal(
            obs.nodes.coordinates[1],
            state.nodes.coordinates[1],
        )

        # Check that the vehicles_positions array is correct
        assert np.array_equal(
            new_state.vehicles.positions,
            jax.numpy.array([1, 2], dtype=jax.numpy.int16),
        )

        # Check that the node demands and action masks are correct.
        assert obs.nodes.demands.shape == (multicvrp_env._num_customers + 1,)
        assert obs.action_mask.shape == (
            multicvrp_env._num_vehicles,
            multicvrp_env._num_customers + 1,
        )

        # Generate a timestep
        is_done = False

        reward = reward_fn(state, new_state, is_done)
        timestep = state_to_timestep_fn(new_state, reward, is_done)

        # Check that the timestep is mid
        assert timestep.mid()

        # Check that the reward and discount values are correct
        assert np.array_equal(timestep.reward, jax.numpy.array(0.0, dtype=jax.numpy.float32))
        assert np.array_equal(timestep.discount, jax.numpy.array(1.0, dtype=jax.numpy.float32))

    def test_env_multicvrp__does_not_smoke(self, multicvrp_env: MultiCVRP) -> None:
        def select_actions(key: chex.PRNGKey, observation: Observation) -> chex.Array:
            @jax.vmap  # map over the agents
            def select_action(key: chex.PRNGKey, agent_action_mask: chex.Array) -> chex.Array:
                return jax.numpy.array(
                    jax.random.choice(
                        key,
                        jax.numpy.arange(len(agent_action_mask)),
                        p=agent_action_mask.flatten(),
                    ),
                    dtype=np.int16,
                )

            subkeys = jax.random.split(key, multicvrp_env._num_vehicles)
            return select_action(subkeys, observation.action_mask)

        check_env_does_not_smoke(multicvrp_env, select_actions)

    def test_env_multicvrp__specs_does_not_smoke(self, multicvrp_env: MultiCVRP) -> None:
        """Test that we can access specs without any errors."""
        check_env_specs_does_not_smoke(multicvrp_env)
