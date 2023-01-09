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
from jumanji.types import TimeStep


def test_env__reset(job_shop_env: JobShop) -> None:
    """Test that the environment is reset correctly."""
    key = jax.random.PRNGKey(0)
    state, timestep = job_shop_env.reset(key)

    assert jnp.all(
        state.operations_machine_ids
        == jnp.array(
            [
                [0, 1, 2],
                [0, 2, 1],
                [1, 2, -1],
            ]
        )
    )
    assert jnp.all(
        state.operations_durations
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
        state.operations_mask
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


def test_env__reset_jitting(job_shop_env: JobShop) -> None:
    """Confirm that the reset is only jitted once."""
    chex.clear_trace_counter()
    reset_fn = jax.jit(chex.assert_max_traces(job_shop_env.reset, n=1))
    key = jax.random.PRNGKey(0)
    state, timestep = reset_fn(key)

    # Call again to check it does not compile twice
    state, timestep = reset_fn(key)
    assert isinstance(timestep, TimeStep)
    assert isinstance(state, State)
