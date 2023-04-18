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
from typing import Any, Tuple

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


class DenseGenerator(Generator):
    """`Generator` which creates a dense schedule of a specified makespan. This is done by:
        - Specifying the `makespan` (schedule length) and the `num_machines`.
        - Initialising an "empty" schedule.
        - Creating a valid schedule:
            1. Randomly sample `num_machines` jobs w/o replacement. These jobs will be
                scheduled on the machines in the first time step.
            2. At the next timestep, stochastically either:
                - Reuse the previous jobs on the machines, or
                - Randomly sample `num_machines` new jobs w/o replacement.
            3. Repeat step 2 until the desired `makespan` is reached.
        - Extracting the info (duration and machine) about operations from the schedule and
         padding the operations to the max number of operations.

    This generator assumes that the number of jobs is less than or equal to the number of
    machines.
    """

    def __init__(
        self,
        num_jobs: int,
        num_machines: int,
        max_num_ops: int,
        max_op_duration: int,
        makespan: int,
    ):
        """Instantiate a `DenseGenerator`. Note that the `makespan` is an upper
        bound to both `max_num_ops` and `max_op_duration`, hence they are not used.

        Args:
            num_jobs: the number of jobs that need to be scheduled.
            num_machines: the number of machines that the jobs can be scheduled on.
            max_num_ops: the maximum number of operations for any given job.
            max_op_duration: the maximum processing time of any given operation.
            makespan: the length of the schedule. By construction, this will be the
                shortest possible length of the schedule.
        """
        del max_op_duration
        del max_num_ops
        if num_jobs < num_machines:
            raise ValueError(
                "The number of jobs must be greater than or equal to the number of machines."
            )

        super().__init__(
            num_jobs=num_jobs,
            num_machines=num_machines,
            max_num_ops=makespan,
            max_op_duration=makespan,
        )
        self.makespan = makespan

    def __call__(self, key: chex.PRNGKey) -> State:
        key, schedule_key = jax.random.split(key)

        # Generate a random, dense schedule of the specified length
        schedule = self._generate_schedule(schedule_key)

        # Extract ops information from the schedule
        ops_machine_ids, ops_durations = self._register_ops(schedule)

        # Initially, all machines are available (the value self.num_jobs corresponds to no-op)
        machines_job_ids = jnp.full(self.num_machines, self.num_jobs, jnp.int32)
        machines_remaining_times = jnp.full(self.num_machines, 0, jnp.int32)
        scheduled_times = jnp.full((self.num_jobs, self.max_num_ops), -1, jnp.int32)
        ops_mask = ops_machine_ids != -1
        step_count = jnp.int32(0)

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

    def _generate_schedule(self, key: chex.PRNGKey) -> chex.Array:
        """Creates a schedule given the constraints of the job shop scheduling problem.

        For example, for 3 machines, 5 jobs, and a chosen optimal makespan of 12, a schedule
        may look like:
        [[1, 0, 1, 1, 2, 3, 4, 0, 1, 3, 2, 3],
         [4, 1, 2, 0, 3, 4, 2, 3, 2, 2, 3, 2],
         [0, 2, 3, 4, 0, 0, 0, 2, 3, 1, 0, 0]]

        This means
            - Machine 0 processes job 1, job 0, job 1 (for two steps), etc.
            - Machine 1 processes job 4, job 1, job 2, job 0, etc.
            - Machine 2 processes job 0, job 2, job 3, job 4, etc.

        Importantly, since a job can only be executed on one machine at a time, this method
        is written such that the schedule does not have duplicates in any column.

        Args:
            key: used for stochasticity in the generation of the schedule.

        Returns:
            Schedule with the specified length. Shape (num_machines, makespan).
        """
        all_job_ids = jnp.arange(self.num_jobs)

        def insert_col(
            carry: Tuple[chex.PRNGKey, int, chex.Array], _: Any
        ) -> Tuple[Tuple[chex.PRNGKey, int, chex.Array], chex.Array]:
            """Create a column of distinct job ids (only one operation in a given job
            can be processed at a time). For the example above, this would be [1, 4, 0] at
            time=0, [0, 1, 2] at time=1, etc.
            """
            key, t, prev_col = carry
            key, job_key, reuse_key = jax.random.split(key, num=3)

            def reuse_prev_col(key: chex.PRNGKey, prev_col: chex.Array) -> chex.Array:
                def _maybe_reuse_op(
                    _carry: Tuple[chex.PRNGKey, chex.Array, int], _: Any
                ) -> Tuple[Tuple[chex.PRNGKey, chex.Array, int], chex.Array]:
                    _key, _job_mask, _machine_id = _carry
                    _key, reuse_op_key, _job_key = jax.random.split(_key, num=3)

                    prev_job_id = prev_col[_machine_id]
                    _job_mask = _job_mask.at[prev_job_id].set(True)

                    # Reuse the previous job with probability 0.7
                    reuse_op = jax.random.uniform(reuse_op_key, shape=()) > 0.3
                    job_id = jax.lax.cond(
                        reuse_op,
                        lambda _: prev_col[_machine_id],
                        lambda _: jax.random.choice(
                            _job_key, all_job_ids, (), p=_job_mask
                        ),
                        None,
                    )
                    _job_mask = _job_mask.at[job_id].set(False)
                    return (_key, _job_mask, _machine_id + 1), job_id

                # Define initial conditions for the scan
                # init_job_mask = jax.vmap(
                #     lambda _job_id, _col: ~jnp.any(_col == _job_id),
                #     in_axes=(0, None),
                # )(all_job_ids, prev_col)
                init_job_mask = jnp.ones(self.num_jobs, dtype=jnp.bool_)
                init_machine_id = 0
                init_carry = (key, init_job_mask, init_machine_id)
                _, col = jax.lax.scan(
                    lambda _carry, _: _maybe_reuse_op(_carry, _),
                    init_carry,
                    xs=None,
                    length=self.num_machines,
                )

                return col

            def sample_new_jobs(key: chex.PRNGKey) -> chex.Array:
                return jax.random.choice(
                    key, all_job_ids, (self.num_machines,), replace=False
                )

            # Reuse the previous column with probability 0.6
            reuse = jax.random.uniform(reuse_key, shape=()) > 0.4
            col = jax.lax.cond(
                reuse,
                lambda _key, _prev_col: reuse_prev_col(_key, _prev_col),
                lambda _key, _: sample_new_jobs(_key),
                job_key,
                prev_col,
            )
            carry = key, t + 1, col
            return carry, col

        init_col = jax.random.choice(
            key,
            all_job_ids,
            (self.num_machines,),
            replace=False,
        )
        init_carry = (key, 0, init_col)
        final_carry, schedule_transposed = jax.lax.scan(
            lambda carry, _: insert_col(carry, _),
            init_carry,
            xs=None,
            length=self.makespan,
        )
        schedule = schedule_transposed.T
        return schedule

    def _register_ops(self, schedule: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Extract, for every job, the machine id and duration of each operation in the job.

        For example, for the schedule
        [[1, 0, 1, 1, 2, 3, 4, 0, 1, 3, 2, 3],
         [4, 1, 2, 0, 3, 4, 2, 3, 2, 2, 3, 2],
         [0, 2, 3, 4, 0, 0, 0, 2, 3, 1, 0, 0]]

        the ops would have the machine ids:
        [[ 2,  0,  1,  2,  0,  2, -1, -1, -1, -1],
         [ 0,  1,  0,  0,  2, -1, -1, -1, -1, -1],
         [ 2,  1,  0,  1,  2,  1,  0,  1, -1, -1],
         [ 2,  1,  0,  1,  2,  0,  1,  0, -1, -1],
         [ 1,  2,  1,  0, -1, -1, -1, -1, -1, -1]]

        and the durations:
        [[ 1,  1,  1,  3,  1,  2, -1, -1, -1, -1],
         [ 1,  1,  2,  1,  1, -1, -1, -1, -1, -1],
         [ 1,  1,  1,  1,  1,  2,  1,  1, -1, -1],
         [ 1,  1,  1,  1,  1,  1,  1,  1, -1, -1],
         [ 1,  1,  1,  1, -1, -1, -1, -1, -1, -1]]

        Args:
            schedule: array representing which job each machine is working on at each timestep.

        Returns:
            Arrays representing which machine id and duration characterising each operation.
        """

        def get_job_info(
            job_id: int, _: Any
        ) -> Tuple[int, Tuple[chex.Array, chex.Array]]:
            """Extract the machine id and duration of every op in the specified job.

            In the above example, for job 0, this will return
                - machine_ids [2,  0,  1,  2,  0,  2, -1, -1, -1, -1]
                - durations [1,  1,  1,  3,  1,  2, -1, -1, -1, -1]
            """

            def get_op_info(
                mask: chex.Array,
                _: Any,
            ) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
                """Extract the machine id and duration for a given operation.

                In the above example, for job 0 and operation 0, the machine id is 2 and
                the duration is 1.

                Args:
                    mask: array which keeps track of which operations have been registered.
                """
                prev_mask = mask

                # Flatten by column
                mask_flat = jnp.ravel(mask, order="F")

                # Find index of the next operation
                idx = jnp.argmax(mask_flat)
                t_start, machine_id = jnp.divmod(idx, self.num_machines)

                # Update the mask -> the op is registered
                mask = mask.at[machine_id, t_start].set(False)

                # While loop in case op has duration > 1
                init_val = (mask, machine_id, t_start + 1)

                def next_is_same_op(val: Tuple) -> chex.Array:
                    m, mid, t = val
                    return m[mid, t]

                def update_mask(val: Tuple) -> Tuple:
                    m, mid, t = val
                    m = m.at[mid, t].set(False)
                    return m, mid, t + 1

                (mask, machine_id, time) = jax.lax.while_loop(
                    next_is_same_op, update_mask, init_val
                )

                duration = time - t_start

                # If all ops for this job are registered, return -1 for padding
                all_ops_registered = ~jnp.any(prev_mask)
                machine_id = jax.lax.select(all_ops_registered, -1, machine_id)
                duration = jax.lax.select(all_ops_registered, -1, duration)

                return mask, (machine_id, duration)

            # Carry the mask
            init_mask = jnp.array(schedule == job_id)
            _, (mids, durs) = jax.lax.scan(
                get_op_info, init_mask, xs=None, length=self.makespan
            )

            return job_id + 1, (mids, durs)

        # Carry the job id
        init_job_id = 0
        job_id, (ops_mids, ops_durs) = jax.lax.scan(
            get_job_info, init_job_id, xs=None, length=self.num_jobs
        )
        return ops_mids, ops_durs
