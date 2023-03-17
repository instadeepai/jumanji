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

import abc

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.packing.job_shop.types import State


class Generator(abc.ABC):
    """Defines the abstract `Generator` base class. A `Generator` is responsible
    for generating a problem instance when the environment is reset.
    """

    def __init__(
        self,
        num_jobs: int,
        num_machines: int,
        max_num_ops: int,
        max_op_duration: int,
    ):
        """Abstract class implementing the attributes `num_jobs`, `num_machines`, `max_num_ops`,
         and `max_op_duration`.

        Args:
            num_jobs: the number of jobs that need to be scheduled.
            num_machines: the number of machines that the jobs can be scheduled on.
            max_num_ops: the maximum number of operations for any given job.
            max_op_duration: the maximum processing time of any given operation.
        """
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.max_num_ops = max_num_ops
        self.max_op_duration = max_op_duration

    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey) -> State:
        """Call method responsible for generating a new state.

        Args:
            key: jax random key in case stochasticity is used in the instance generation process.

        Returns:
            A `JobShop` environment state.
        """


class ToyGenerator(Generator):
    """`Generator` that can be used as an example. It deterministically outputs a hardcoded
    instance with 5 jobs, 4 machines, a max of 4 ops for any job, and max duration of 4 time
    steps for any operation. By construction, this generator has a known, optimal makespan
    of 8 time steps.
    """

    def __init__(self) -> None:
        super().__init__(num_jobs=5, num_machines=4, max_num_ops=4, max_op_duration=4)

    def __call__(self, key: chex.PRNGKey) -> State:
        del key

        ops_machine_ids = jnp.array(
            [
                [2, 3, 1, 2],
                [3, 2, 0, -1],
                [1, 3, -1, -1],
                [0, 3, 0, 0],
                [1, 0, 1, -1],
            ],
            jnp.int32,
        )
        ops_durations = jnp.array(
            [
                [2, 2, 1, 2],
                [2, 4, 1, -1],
                [2, 3, -1, -1],
                [4, 1, 1, 1],
                [3, 1, 2, -1],
            ],
            jnp.int32,
        )

        # Initially, all machines are available (the value self.num_jobs corresponds to no-op)
        machines_job_ids = jnp.full(self.num_machines, self.num_jobs, jnp.int32)
        machines_remaining_times = jnp.full(self.num_machines, 0, jnp.int32)
        scheduled_times = jnp.full((self.num_jobs, self.max_num_ops), -1, jnp.int32)
        ops_mask = ops_machine_ids != -1
        step_count = jnp.array(0, jnp.int32)

        state = State(
            ops_machine_ids=ops_machine_ids,
            ops_durations=ops_durations,
            ops_mask=ops_mask,
            machines_job_ids=machines_job_ids,
            machines_remaining_times=machines_remaining_times,
            action_mask=None,
            step_count=step_count,
            scheduled_times=scheduled_times,
            key=jax.random.PRNGKey(0),
        )

        return state


class RandomGenerator(Generator):
    """Instance generator that generates random instances of the job shop scheduling problem. Given
    the number of machines, number of jobs, max number of operations for any job, and max duration
    of any operation, the generation works as follows: for each job, we sample the number of ops
    for that job. Then, for each operation, a machine_id and duration are sampled, both from random
    uniform distributions. Finally, padding is done for jobs whose number of operations is less than
    the max.
    """

    def __init__(
        self, num_jobs: int, num_machines: int, max_num_ops: int, max_op_duration: int
    ):
        super().__init__(num_jobs, num_machines, max_num_ops, max_op_duration)

    def __call__(self, key: chex.PRNGKey) -> State:
        key, machine_key, duration_key, ops_key = jax.random.split(key, num=4)

        # Randomly sample machine IDs and durations
        ops_machine_ids = jax.random.randint(
            machine_key,
            shape=(self.num_jobs, self.max_num_ops),
            minval=0,
            maxval=self.num_machines,
        )
        ops_durations = jax.random.randint(
            duration_key,
            shape=(self.num_jobs, self.max_num_ops),
            minval=1,
            maxval=self.max_op_duration + 1,
        )

        # Vary the number of ops across jobs
        num_ops_per_job = jax.random.randint(
            ops_key,
            shape=(self.num_jobs,),
            minval=1,
            maxval=self.max_num_ops + 1,
        )
        mask = jnp.less(
            jnp.tile(jnp.arange(self.max_num_ops), reps=(self.num_jobs, 1)),
            jnp.expand_dims(num_ops_per_job, axis=-1),
        )
        ops_machine_ids = jnp.where(mask, ops_machine_ids, jnp.array(-1, jnp.int32))
        ops_durations = jnp.where(mask, ops_durations, jnp.array(-1, jnp.int32))

        # Initially, all machines are available (the value self.num_jobs corresponds to no-op)
        machines_job_ids = jnp.full(self.num_machines, self.num_jobs, jnp.int32)
        machines_remaining_times = jnp.full(self.num_machines, 0, jnp.int32)

        # Initially, none of the operations have been scheduled
        scheduled_times = jnp.full((self.num_jobs, self.max_num_ops), -1, jnp.int32)
        ops_mask = ops_machine_ids != -1

        # Time starts at 0
        step_count = jnp.array(0, jnp.int32)

        state = State(
            ops_machine_ids=ops_machine_ids,
            ops_durations=ops_durations,
            ops_mask=ops_mask,
            machines_job_ids=machines_job_ids,
            machines_remaining_times=machines_remaining_times,
            action_mask=None,
            step_count=step_count,
            scheduled_times=scheduled_times,
            key=key,
        )

        return state
