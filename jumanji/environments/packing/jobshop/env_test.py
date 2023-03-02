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
import jax.numpy as jnp

from jumanji.environments.packing.jobshop.env import JobShop
from jumanji.environments.packing.jobshop.types import State
from jumanji.testing.env_not_smoke import check_env_does_not_smoke
from jumanji.types import TimeStep


class TestJobShop:
    def test_env__reset(self, job_shop_env: JobShop) -> None:
        """Test that the environment is reset correctly."""
        key = jax.random.PRNGKey(0)
        state, timestep = job_shop_env.reset(key)

        assert jnp.all(
            state.ops_machine_ids
            == jnp.array(
                [
                    [0, 1, 2],
                    [0, 2, 1],
                    [1, 2, -1],
                ]
            )
        )
        assert jnp.all(
            state.ops_durations
            == jnp.array(
                [
                    [3, 2, 2],
                    [2, 1, 4],
                    [4, 3, -1],
                ]
            )
        )
        assert jnp.all(state.machines_job_ids == jnp.array([3, 3, 3]))
        assert jnp.all(state.machines_remaining_times == jnp.array([0, 0, 0]))
        assert jnp.all(
            state.action_mask
            == jnp.array(
                [
                    [True, True, False, True],
                    [False, False, True, True],
                    [False, False, False, True],
                ]
            )
        )
        assert jnp.all(
            state.ops_mask
            == jnp.array([[True, True, True], [True, True, True], [True, True, False]])
        )
        assert jnp.all(
            state.scheduled_times
            == jnp.array(
                [
                    [-1, -1, -1],
                    [-1, -1, -1],
                    [-1, -1, -1],
                ],
            )
        )
        assert state.current_timestep == jnp.int32(0)

    def test_env__reset_jit(self, job_shop_env: JobShop) -> None:
        """Confirm that the reset is only compiled once when jitted."""
        chex.clear_trace_counter()
        reset_fn = jax.jit(chex.assert_max_traces(job_shop_env.reset, n=1))
        key = jax.random.PRNGKey(0)
        state, timestep = reset_fn(key)

        # Call again to check it does not compile twice
        state, timestep = reset_fn(key)
        assert isinstance(timestep, TimeStep)
        assert isinstance(state, State)

    def test_env__step(self, job_shop_env: JobShop) -> None:
        """Test the 12 steps of the dummy instance."""

        key = jax.random.PRNGKey(0)
        state, timestep = job_shop_env.reset(key)

        # STEP T=0 -> T=1
        action = jnp.array([1, 2, 3])
        next_state, next_timestep = job_shop_env.step(state, action)

        assert jnp.all(
            next_state.ops_machine_ids
            == jnp.array(
                [
                    [0, 1, 2],
                    [0, 2, 1],
                    [1, 2, -1],
                ]
            )
        )
        assert jnp.all(
            next_state.ops_durations
            == jnp.array(
                [
                    [3, 2, 2],
                    [2, 1, 4],
                    [4, 3, -1],
                ]
            )
        )
        assert jnp.all(next_state.machines_remaining_times == jnp.array([1, 3, 0]))
        assert jnp.all(next_state.machines_job_ids == jnp.array([1, 2, 3]))
        assert jnp.all(
            next_state.action_mask
            == jnp.array(
                [
                    [False, False, False, True],
                    [False, False, False, True],
                    [False, False, False, True],
                ]
            )
        )
        assert jnp.all(
            next_state.ops_mask
            == jnp.array(
                [[True, True, True], [False, True, True], [False, True, False]]
            )
        )
        assert jnp.all(
            next_state.scheduled_times
            == jnp.array(
                [
                    [-1, -1, -1],
                    [0, -1, -1],
                    [0, -1, -1],
                ]
            )
        )
        assert next_state.current_timestep == 1

        # STEP T=1 -> T=2
        next_action = jnp.array([3, 3, 3])
        next_state, next_timestep = job_shop_env.step(
            state=next_state, action=next_action
        )
        assert jnp.all(
            next_state.ops_machine_ids
            == jnp.array(
                [
                    [0, 1, 2],
                    [0, 2, 1],
                    [1, 2, -1],
                ]
            )
        )
        assert jnp.all(
            next_state.ops_durations
            == jnp.array(
                [
                    [3, 2, 2],
                    [2, 1, 4],
                    [4, 3, -1],
                ]
            )
        )
        assert jnp.all(next_state.machines_job_ids == jnp.array([1, 2, 3]))
        assert jnp.all(next_state.machines_remaining_times == jnp.array([0, 2, 0]))
        assert jnp.all(
            next_state.action_mask
            == jnp.array(
                [
                    [True, False, False, True],
                    [False, False, False, True],
                    [False, True, False, True],
                ]
            )
        )
        assert jnp.all(
            next_state.ops_mask
            == jnp.array(
                [[True, True, True], [False, True, True], [False, True, False]]
            )
        )
        assert jnp.all(
            next_state.scheduled_times
            == jnp.array(
                [
                    [-1, -1, -1],
                    [0, -1, -1],
                    [0, -1, -1],
                ]
            )
        )
        assert next_state.current_timestep == 2

        # STEP T=2 -> T=3
        next_action = jnp.array([0, 3, 1])
        next_state, next_timestep = job_shop_env.step(
            state=next_state, action=next_action
        )

        assert jnp.all(
            next_state.ops_machine_ids
            == jnp.array(
                [
                    [0, 1, 2],
                    [0, 2, 1],
                    [1, 2, -1],
                ]
            )
        )
        assert jnp.all(
            next_state.ops_durations
            == jnp.array(
                [
                    [3, 2, 2],
                    [2, 1, 4],
                    [4, 3, -1],
                ]
            )
        )
        assert jnp.all(next_state.machines_job_ids == jnp.array([0, 2, 1]))
        assert jnp.all(next_state.machines_remaining_times == jnp.array([2, 1, 0]))
        assert jnp.all(
            next_state.action_mask
            == jnp.array(
                [
                    [False, False, False, True],
                    [False, False, False, True],
                    [False, False, False, True],
                ]
            )
        )
        assert jnp.all(
            next_state.ops_mask
            == jnp.array(
                [[False, True, True], [False, False, True], [False, True, False]]
            )
        )
        assert jnp.all(
            next_state.scheduled_times
            == jnp.array(
                [
                    [2, -1, -1],
                    [0, 2, -1],
                    [0, -1, -1],
                ]
            )
        )
        assert next_state.current_timestep == 3

        # STEP T=3 -> T=4
        next_action = jnp.array([3, 3, 3])
        next_state, next_timestep = job_shop_env.step(
            state=next_state, action=next_action
        )

        assert jnp.all(
            next_state.ops_machine_ids
            == jnp.array(
                [
                    [0, 1, 2],
                    [0, 2, 1],
                    [1, 2, -1],
                ]
            )
        )
        assert jnp.all(
            next_state.ops_durations
            == jnp.array(
                [
                    [3, 2, 2],
                    [2, 1, 4],
                    [4, 3, -1],
                ]
            )
        )
        assert jnp.all(next_state.machines_job_ids == jnp.array([0, 2, 3]))
        assert jnp.all(next_state.machines_remaining_times == jnp.array([1, 0, 0]))
        assert jnp.all(
            next_state.action_mask
            == jnp.array(
                [
                    [False, False, False, True],
                    [False, True, False, True],
                    [False, False, True, True],
                ]
            )
        )
        assert jnp.all(
            next_state.ops_mask
            == jnp.array(
                [[False, True, True], [False, False, True], [False, True, False]]
            )
        )
        assert jnp.all(
            next_state.scheduled_times
            == jnp.array(
                [
                    [2, -1, -1],
                    [0, 2, -1],
                    [0, -1, -1],
                ]
            )
        )
        assert next_state.current_timestep == 4

        # STEP T=4 -> T=5
        next_action = jnp.array([3, 1, 2])
        next_state, next_timestep = job_shop_env.step(
            state=next_state, action=next_action
        )

        assert jnp.all(
            next_state.ops_machine_ids
            == jnp.array(
                [
                    [0, 1, 2],
                    [0, 2, 1],
                    [1, 2, -1],
                ]
            )
        )
        assert jnp.all(
            next_state.ops_durations
            == jnp.array(
                [
                    [3, 2, 2],
                    [2, 1, 4],
                    [4, 3, -1],
                ]
            )
        )
        assert jnp.all(next_state.machines_job_ids == jnp.array([0, 1, 2]))
        assert jnp.all(next_state.machines_remaining_times == jnp.array([0, 3, 2]))
        assert jnp.all(
            next_state.action_mask
            == jnp.array(
                [
                    [False, False, False, True],
                    [False, False, False, True],
                    [False, False, False, True],
                ]
            )
        )
        assert jnp.all(
            next_state.ops_mask
            == jnp.array(
                [[False, True, True], [False, False, False], [False, False, False]]
            )
        )
        assert jnp.all(
            next_state.scheduled_times
            == jnp.array(
                [
                    [2, -1, -1],
                    [0, 2, 4],
                    [0, 4, -1],
                ]
            )
        )
        assert next_state.current_timestep == 5

        # STEP T=5 -> T=6
        next_action = jnp.array([3, 3, 3])
        next_state, next_timestep = job_shop_env.step(
            state=next_state, action=next_action
        )

        assert jnp.all(
            next_state.ops_machine_ids
            == jnp.array(
                [
                    [0, 1, 2],
                    [0, 2, 1],
                    [1, 2, -1],
                ]
            )
        )
        assert jnp.all(
            next_state.ops_durations
            == jnp.array(
                [
                    [3, 2, 2],
                    [2, 1, 4],
                    [4, 3, -1],
                ]
            )
        )
        assert jnp.all(next_state.machines_job_ids == jnp.array([3, 1, 2]))
        assert jnp.all(next_state.machines_remaining_times == jnp.array([0, 2, 1]))
        assert jnp.all(
            next_state.action_mask
            == jnp.array(
                [
                    [False, False, False, True],
                    [False, False, False, True],
                    [False, False, False, True],
                ]
            )
        )
        assert jnp.all(
            next_state.ops_mask
            == jnp.array(
                [[False, True, True], [False, False, False], [False, False, False]]
            )
        )
        assert jnp.all(
            next_state.scheduled_times
            == jnp.array(
                [
                    [2, -1, -1],
                    [0, 2, 4],
                    [0, 4, -1],
                ]
            )
        )
        assert next_state.current_timestep == 6

        # STEP T=6 -> T=7
        next_action = jnp.array([3, 3, 3])
        next_state, next_timestep = job_shop_env.step(
            state=next_state, action=next_action
        )

        assert jnp.all(
            next_state.ops_machine_ids
            == jnp.array(
                [
                    [0, 1, 2],
                    [0, 2, 1],
                    [1, 2, -1],
                ]
            )
        )
        assert jnp.all(
            next_state.ops_durations
            == jnp.array(
                [
                    [3, 2, 2],
                    [2, 1, 4],
                    [4, 3, -1],
                ]
            )
        )
        assert jnp.all(next_state.machines_job_ids == jnp.array([3, 1, 2]))
        assert jnp.all(next_state.machines_remaining_times == jnp.array([0, 1, 0]))
        assert jnp.all(
            next_state.action_mask
            == jnp.array(
                [
                    [False, False, False, True],
                    [False, False, False, True],
                    [False, False, False, True],
                ]
            )
        )
        assert jnp.all(
            next_state.ops_mask
            == jnp.array(
                [[False, True, True], [False, False, False], [False, False, False]]
            )
        )
        assert jnp.all(
            next_state.scheduled_times
            == jnp.array(
                [
                    [2, -1, -1],
                    [0, 2, 4],
                    [0, 4, -1],
                ]
            )
        )
        assert next_state.current_timestep == 7

        # STEP T=7 -> T=8
        next_action = jnp.array([3, 3, 3])
        next_state, next_timestep = job_shop_env.step(
            state=next_state, action=next_action
        )

        assert jnp.all(
            next_state.ops_machine_ids
            == jnp.array(
                [
                    [0, 1, 2],
                    [0, 2, 1],
                    [1, 2, -1],
                ]
            )
        )
        assert jnp.all(
            next_state.ops_durations
            == jnp.array(
                [
                    [3, 2, 2],
                    [2, 1, 4],
                    [4, 3, -1],
                ]
            )
        )
        assert jnp.all(next_state.machines_job_ids == jnp.array([3, 1, 3]))
        assert jnp.all(next_state.machines_remaining_times == jnp.array([0, 0, 0]))
        assert jnp.all(
            next_state.action_mask
            == jnp.array(
                [
                    [False, False, False, True],
                    [True, False, False, True],
                    [False, False, False, True],
                ]
            )
        )
        assert jnp.all(
            next_state.ops_mask
            == jnp.array(
                [[False, True, True], [False, False, False], [False, False, False]]
            )
        )
        assert jnp.all(
            next_state.scheduled_times
            == jnp.array(
                [
                    [2, -1, -1],
                    [0, 2, 4],
                    [0, 4, -1],
                ]
            )
        )
        assert next_state.current_timestep == 8

        # STEP T=8 -> T=9
        next_action = jnp.array([3, 0, 3])
        next_state, next_timestep = job_shop_env.step(
            state=next_state, action=next_action
        )

        assert jnp.all(
            next_state.ops_machine_ids
            == jnp.array(
                [
                    [0, 1, 2],
                    [0, 2, 1],
                    [1, 2, -1],
                ]
            )
        )
        assert jnp.all(
            next_state.ops_durations
            == jnp.array(
                [
                    [3, 2, 2],
                    [2, 1, 4],
                    [4, 3, -1],
                ]
            )
        )
        assert jnp.all(next_state.machines_job_ids == jnp.array([3, 0, 3]))
        assert jnp.all(next_state.machines_remaining_times == jnp.array([0, 1, 0]))
        assert jnp.all(
            next_state.action_mask
            == jnp.array(
                [
                    [False, False, False, True],
                    [False, False, False, True],
                    [False, False, False, True],
                ]
            )
        )
        assert jnp.all(
            next_state.ops_mask
            == jnp.array(
                [[False, False, True], [False, False, False], [False, False, False]]
            )
        )
        assert jnp.all(
            next_state.scheduled_times
            == jnp.array(
                [
                    [2, 8, -1],
                    [0, 2, 4],
                    [0, 4, -1],
                ]
            )
        )
        assert next_state.current_timestep == 9

        # # STEP T=9 -> T=10
        next_action = jnp.array([3, 3, 3])
        next_state, next_timestep = job_shop_env.step(
            state=next_state, action=next_action
        )

        assert jnp.all(
            next_state.ops_machine_ids
            == jnp.array(
                [
                    [0, 1, 2],
                    [0, 2, 1],
                    [1, 2, -1],
                ]
            )
        )
        assert jnp.all(
            next_state.ops_durations
            == jnp.array(
                [
                    [3, 2, 2],
                    [2, 1, 4],
                    [4, 3, -1],
                ]
            )
        )
        assert jnp.all(next_state.machines_job_ids == jnp.array([3, 0, 3]))
        assert jnp.all(next_state.machines_remaining_times == jnp.array([0, 0, 0]))
        assert jnp.all(
            next_state.action_mask
            == jnp.array(
                [
                    [False, False, False, True],
                    [False, False, False, True],
                    [True, False, False, True],
                ]
            )
        )
        assert jnp.all(
            next_state.ops_mask
            == jnp.array(
                [[False, False, True], [False, False, False], [False, False, False]]
            )
        )
        assert jnp.all(
            next_state.scheduled_times
            == jnp.array(
                [
                    [2, 8, -1],
                    [0, 2, 4],
                    [0, 4, -1],
                ]
            )
        )
        assert next_state.current_timestep == 10

        # # STEP T=10 -> T=11
        next_action = jnp.array([3, 3, 0])
        next_state, next_timestep = job_shop_env.step(
            state=next_state, action=next_action
        )

        assert jnp.all(
            next_state.ops_machine_ids
            == jnp.array(
                [
                    [0, 1, 2],
                    [0, 2, 1],
                    [1, 2, -1],
                ]
            )
        )
        assert jnp.all(
            next_state.ops_durations
            == jnp.array(
                [
                    [3, 2, 2],
                    [2, 1, 4],
                    [4, 3, -1],
                ]
            )
        )
        assert jnp.all(next_state.machines_job_ids == jnp.array([3, 3, 0]))
        assert jnp.all(next_state.machines_remaining_times == jnp.array([0, 0, 1]))
        assert jnp.all(
            next_state.action_mask
            == jnp.array(
                [
                    [False, False, False, True],
                    [False, False, False, True],
                    [False, False, False, True],
                ]
            )
        )
        assert jnp.all(
            next_state.ops_mask
            == jnp.array(
                [[False, False, False], [False, False, False], [False, False, False]]
            )
        )
        assert jnp.all(
            next_state.scheduled_times
            == jnp.array(
                [
                    [2, 8, 10],
                    [0, 2, 4],
                    [0, 4, -1],
                ]
            )
        )
        assert next_state.current_timestep == 11

        # STEP T=10 -> T=11
        next_action = jnp.array([3, 3, 3])
        next_state, next_timestep = job_shop_env.step(
            state=next_state, action=next_action
        )

        assert jnp.all(
            next_state.ops_machine_ids
            == jnp.array(
                [
                    [0, 1, 2],
                    [0, 2, 1],
                    [1, 2, -1],
                ]
            )
        )
        assert jnp.all(
            next_state.ops_durations
            == jnp.array(
                [
                    [3, 2, 2],
                    [2, 1, 4],
                    [4, 3, -1],
                ]
            )
        )
        assert jnp.all(next_state.machines_job_ids == jnp.array([3, 3, 0]))
        assert jnp.all(next_state.machines_remaining_times == jnp.array([0, 0, 0]))
        assert jnp.all(
            next_state.action_mask
            == jnp.array(
                [
                    [False, False, False, True],
                    [False, False, False, True],
                    [False, False, False, True],
                ]
            )
        )
        assert jnp.all(
            next_state.ops_mask
            == jnp.array(
                [[False, False, False], [False, False, False], [False, False, False]]
            )
        )
        assert jnp.all(
            next_state.scheduled_times
            == jnp.array(
                [
                    [2, 8, 10],
                    [0, 2, 4],
                    [0, 4, -1],
                ]
            )
        )
        assert next_state.current_timestep == 12

    def test_env__step_jit(self, job_shop_env: JobShop) -> None:
        """Confirm that the step is only compiled once when jitted."""
        key = jax.random.PRNGKey(0)
        state, timestep = job_shop_env.reset(key)
        action = jnp.array([1, 2, 3], jnp.int32)

        chex.clear_trace_counter()
        step_fn = jax.jit(chex.assert_max_traces(job_shop_env.step, n=1))
        _, _ = step_fn(state, action)

        # Call again to check it does not compile twice
        next_state, next_timestep = step_fn(state, action)
        assert isinstance(next_timestep, TimeStep)
        assert isinstance(next_state, State)

    def test_jobshop_env__does_not_smoke(self, job_shop_env: JobShop) -> None:
        """Test that we can run an episode without any errors."""
        check_env_does_not_smoke(job_shop_env)
