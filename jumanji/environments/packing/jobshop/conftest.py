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

import jax.numpy as jnp
import pytest
from chex import PRNGKey

from jumanji.environments.packing.jobshop.env import JobShop
from jumanji.environments.packing.jobshop.instance_generator import InstanceGenerator
from jumanji.environments.packing.jobshop.types import State


class DummyInstanceGenerator(InstanceGenerator):
    """Hardcoded `InstanceGenerator` mainly used for testing and debugging. It deterministically
    outputs a hardcoded instance with 3 jobs, 3 machines, a max of 3 ops for any job, and a max
    duration of 4 time steps for any operation.
    """

    def __init__(self) -> None:
        super().__init__(
            num_jobs=3,
            num_machines=3,
            max_num_ops=3,
            max_op_duration=4,
        )

    def __call__(self, key: PRNGKey) -> State:
        """Call method responsible for generating a new state. It returns a job shop scheduling
        instance without any scheduled jobs.

        Args:
            key: jax random key for any stochasticity used in the generation process. Not used
                in this instance generator.

        Returns:
            A JobShop State.
        """
        del key

        operations_machine_ids = jnp.array(
            [
                [0, 1, 2],
                [0, 2, 1],
                [1, 2, -1],
            ],
            jnp.int32,
        )
        operations_durations = jnp.array(
            [
                [3, 2, 2],
                [2, 1, 4],
                [4, 3, -1],
            ],
            jnp.int32,
        )

        # Initially, all machines are available (the index self.num_jobs corresponds to no-op)
        machines_job_ids = jnp.array(
            [self.num_jobs, self.num_jobs, self.num_jobs], jnp.int32
        )
        machines_remaining_times = jnp.array([0, 0, 0], jnp.int32)

        # Initial action mask given the problem instance
        action_mask = jnp.array(
            [
                [True, True, False, True],  # Machine 0 legal actions: Job0/Job1/No-op
                [False, False, True, True],  # Machine 1 legal actions: Job2/No-op
                [False, False, False, True],  # Machine 2 legal actions: No-op
            ],
            bool,
        )

        # Initially, all ops have yet to be scheduled (ignore the padded element)
        operations_mask = jnp.array(
            [[True, True, True], [True, True, True], [True, True, False]], bool
        )

        # Initially, none of the operations have been scheduled
        scheduled_times = jnp.array(
            [
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
            ],
            jnp.int32,
        )

        current_timestep = jnp.int32(0)

        state = State(
            operations_machine_ids=operations_machine_ids,
            operations_durations=operations_durations,
            operations_mask=operations_mask,
            machines_job_ids=machines_job_ids,
            machines_remaining_times=machines_remaining_times,
            action_mask=action_mask,
            current_timestep=current_timestep,
            scheduled_times=scheduled_times,
        )

        return state


@pytest.fixture
def job_shop_env() -> JobShop:
    env = JobShop()
    env.instance_generator = DummyInstanceGenerator()
    env.num_jobs = env.instance_generator.num_jobs
    env.num_machines = env.instance_generator.num_machines
    env.max_num_ops = env.instance_generator.max_num_ops
    env.max_op_duration = env.instance_generator.max_op_duration
    return env
