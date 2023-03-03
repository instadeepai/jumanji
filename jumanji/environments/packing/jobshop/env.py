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

from typing import Any, Optional, Tuple

import chex
import jax
import jax.numpy as jnp

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.packing.jobshop.env_viewer import JobShopViewer
from jumanji.environments.packing.jobshop.generator import Generator, RandomGenerator
from jumanji.environments.packing.jobshop.types import Observation, State
from jumanji.types import Action, TimeStep, restart, termination, transition


class JobShop(Environment[State]):
    """The Job Shop Scheduling Problem, as described in [1], is one of the best known
    combinatorial optimization problems. We are given `num_jobs` jobs, each consisting
    of at most `max_num_ops` ops, which need to be processed on `num_machines` machines.
    Each operation (op) has a specific machine that it needs to be processed on and a
    duration (which must be less than or equal to `max_duration_op`). The goal is to
    minimise the total length of the schedule, also known as the makespan.

    [1] https://developers.google.com/optimization/scheduling/job_shop.
    """

    def __init__(self, generator: Optional[Generator] = None):
        """Instantiate a `JobShop` environment.

        Args:
            generator: `Generator` whose `__call__` instantiates an environment instance.
                Defaults to `RandomGenerator` with its default parameters.
        """
        self.generator = generator or RandomGenerator()
        self.num_jobs = self.generator.num_jobs
        self.num_machines = self.generator.num_machines
        self.max_num_ops = self.generator.max_num_ops
        self.max_op_duration = self.generator.max_op_duration

        # Define the "job id" of a no-op action as the number of jobs
        self.no_op_idx = self.num_jobs

        # Create viewer used for rendering
        self._env_viewer = JobShopViewer(
            "JobShop",
            self.num_jobs,
            self.num_machines,
            self.max_num_ops,
            self.max_op_duration,
        )

    def __repr__(self) -> str:
        return "\n".join(
            [
                "JobShop environment:",
                f" - generator: {self.generator}",
                f" - num_jobs: {self.num_jobs}",
                f" - num_machines: {self.num_machines}",
                f" - max_num_ops: {self.max_num_ops}",
                f" - max_op_duration: {self.max_op_duration}",
            ]
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment by creating a new problem instance and initialising the state
        and timestep.

        Args:
            key: random key used to reset the environment.

        Returns:
            state: the environment state after the reset.
            timestep: the first timestep returned by the environment after the reset.
        """
        # Generate a new problem instance
        state = self.generator(key)

        # Create the action mask and update the state
        state.action_mask = self._create_action_mask(
            state.machines_job_ids,
            state.machines_remaining_times,
            state.ops_machine_ids,
            state.ops_mask,
        )

        # Get the observation and the timestep
        obs = self._observation_from_state(state)
        timestep = restart(observation=obs)

        return state, timestep

    def step(self, state: State, action: Action) -> Tuple[State, TimeStep[Observation]]:
        """Updates the status of all machines, the status of the operations, and increments the
        time step. It updates the environment state and the timestep (which contains the new
        observation). It calculates the reward based on the three terminal conditions:
            - The action provided by the agent is invalid.
            - The schedule has finished.
            - All machines do a no-op that leads to all machines being simultaneously idle.

        Args:
            state: the environment state.
            action: the action to take.

        Returns:
            state: the updated environment state.
            timestep: the updated timestep.
        """
        # Check the action is legal
        invalid = ~jnp.all(state.action_mask[jnp.arange(self.num_machines), action])  # type: ignore

        # Obtain the id for every job's next operation
        op_ids = jnp.argmax(state.ops_mask, axis=-1)

        # Update the status of all machines
        (
            updated_machines_job_ids,
            updated_machines_remaining_times,
        ) = self._update_machines(
            action,
            op_ids,
            state.machines_job_ids,
            state.machines_remaining_times,
            state.ops_durations,
        )

        # Update the status of operations that have been scheduled
        updated_ops_mask, updated_scheduled_times = self._update_operations(
            action,
            op_ids,
            state.current_timestep,
            state.scheduled_times,
            state.ops_mask,
        )

        # Update the action_mask
        updated_action_mask = self._create_action_mask(
            updated_machines_job_ids,
            updated_machines_remaining_times,
            state.ops_machine_ids,
            updated_ops_mask,
        )

        # Increment the time step
        updated_timestep = jnp.int32(state.current_timestep + 1)

        # Check if all machines are idle simultaneously
        all_machines_idle = jnp.all(
            (updated_machines_job_ids == self.no_op_idx)
            & (updated_machines_remaining_times == 0)
        )

        # Check if the schedule has finished
        all_operations_scheduled = ~jnp.any(updated_ops_mask)
        schedule_finished = all_operations_scheduled & jnp.all(
            updated_machines_remaining_times == 0
        )

        # Update the state and extract the next observation
        next_state = State(
            ops_machine_ids=state.ops_machine_ids,
            ops_durations=state.ops_durations,
            ops_mask=updated_ops_mask,
            machines_job_ids=updated_machines_job_ids,
            machines_remaining_times=updated_machines_remaining_times,
            action_mask=updated_action_mask,
            current_timestep=updated_timestep,
            scheduled_times=updated_scheduled_times,
        )
        next_obs = self._observation_from_state(next_state)

        # Compute terminal condition
        done = invalid | all_machines_idle | schedule_finished
        reward = jnp.where(
            invalid | all_machines_idle,
            jnp.array(-self.num_jobs * self.max_num_ops * self.max_op_duration, float),
            jnp.array(-1, float),
        )

        timestep = jax.lax.cond(
            done,
            lambda _: termination(reward=reward, observation=next_obs),
            lambda _: transition(reward=reward, observation=next_obs),
            operand=None,
        )

        return next_state, timestep

    def _update_operations(
        self,
        action: Action,
        op_ids: chex.Array,
        current_timestep: int,
        scheduled_times: chex.Array,
        ops_mask: chex.Array,
    ) -> Any:
        """Update the operations mask and schedule times based on the newly scheduled
         operations as detailed by the action.

         Args:
            action: array representing the job (or no-op) scheduled by each machine.
            op_ids: array containing the index of the next operation for each job.
            current_timestep: the current time.
            scheduled_times: specifies the time step at which each operation was
                scheduled. Set to -1 if the operation has not been scheduled yet.
            ops_mask: specifies which operations have yet to be scheduled.

        Returns:
            updated_ops_mask: specifies which operations have yet to be scheduled.
            updated_scheduled_times: specifies the time step at which each operation was
                scheduled. Set to -1 if the operation has not been scheduled yet.
        """
        job_id_batched = jnp.arange(self.num_jobs)
        is_new_job = jax.vmap(self._set_busy, in_axes=(0, None))(job_id_batched, action)
        is_new_job = jnp.tile(is_new_job, reps=(self.max_num_ops, 1)).T
        is_next_op = jnp.zeros(shape=(self.num_jobs, self.max_num_ops), dtype=bool)
        is_next_op = is_next_op.at[jnp.arange(self.num_jobs), op_ids].set(True)
        is_new_job_and_next_op = jnp.logical_and(is_new_job, is_next_op)
        updated_scheduled_times = jnp.where(
            is_new_job_and_next_op, current_timestep, scheduled_times
        )
        updated_ops_mask = ops_mask & ~is_new_job_and_next_op
        return updated_ops_mask, updated_scheduled_times

    def _update_machines(
        self,
        action: Action,
        op_ids: chex.Array,
        machines_job_ids: chex.Array,
        machines_remaining_times: chex.Array,
        ops_durations: chex.Array,
    ) -> Any:
        """Update the status of all machines based on the action, specifically:
            - The specific job (or no-op) processed by each machine.
            - The number of time steps until each machine is available.

        Args:
            action: array representing the job (or no-op) scheduled by each machine.
            op_ids: array containing the index of the next operation for each job.
            machines_job_ids: array containing the job (or no-op) processed by each machine.
            machines_remaining_times: array containing the time remaining until available
                for each machine.
            ops_durations: for each job, it specifies the processing time of each
                operation.

        Returns:
            updated_machines_job_ids: array containing the job (or no-op) processed by
                each machine.
            updated_machines_remaining_times: array containing the time remaining
                until available for each machine.
        """
        # For machines that do a no-op, keep the existing job (or no-op)
        job_ids = jnp.where(action == self.no_op_idx, machines_job_ids, action)

        # If a machine does a no-op and is available, schedule the job (or no-op)
        updated_machines_job_ids = jnp.where(
            (action == self.no_op_idx) & (machines_remaining_times == 0),
            action,
            job_ids,
        )

        # If a machine does a no-op, keep the existing remaining time
        # Otherwise, use the duration of the job's operation
        remaining_times = jnp.where(
            action == self.no_op_idx,
            machines_remaining_times,
            ops_durations[action, op_ids[action]],
        )

        # For busy machines, decrement the remaining time by one
        updated_machines_remaining_times = jnp.where(
            remaining_times > 0, remaining_times - 1, 0
        )

        return updated_machines_job_ids, updated_machines_remaining_times

    def observation_spec(self) -> specs.Spec[Observation]:
        """Specifications of the observation of the `JobShop` environment.

        Returns:
            Spec containing the specifications for all the `Observation` fields:
            - ops_machine_ids: BoundedArray (int32) of shape (num_jobs, max_num_ops).
            - ops_durations: BoundedArray (int32) of shape (num_jobs, max_num_ops).
            - ops_mask: BoundedArray (bool) of shape (num_jobs, max_num_ops).
            - machines_job_ids: BoundedArray (int32) of shape (num_machines,).
            - machines_remaining_times: BoundedArray (int32) of shape (num_machines,).
            - action_mask: BoundedArray (bool) of shape (num_machines, num_jobs + 1).
        """
        ops_machine_ids = specs.BoundedArray(
            shape=(self.num_jobs, self.max_num_ops),
            dtype=jnp.int32,
            minimum=-1,
            maximum=self.num_machines - 1,
            name="ops_machine_ids",
        )
        ops_durations = specs.BoundedArray(
            shape=(self.num_jobs, self.max_num_ops),
            dtype=jnp.int32,
            minimum=-1,
            maximum=self.max_op_duration,
            name="ops_durations",
        )
        ops_mask = specs.BoundedArray(
            shape=(self.num_jobs, self.max_num_ops),
            dtype=bool,
            minimum=False,
            maximum=True,
            name="ops_mask",
        )
        machines_job_ids = specs.BoundedArray(
            shape=(self.num_machines,),
            dtype=jnp.int32,
            minimum=0,
            maximum=self.num_jobs,
            name="machines_job_ids",
        )
        machines_remaining_times = specs.BoundedArray(
            shape=(self.num_machines,),
            dtype=jnp.int32,
            minimum=0,
            maximum=self.max_op_duration,
            name="machines_remaining_times",
        )
        action_mask = specs.BoundedArray(
            shape=(self.num_machines, self.num_jobs + 1),
            dtype=bool,
            minimum=False,
            maximum=True,
            name="action_mask",
        )
        return specs.Spec(
            constructor=Observation,
            name="ObservationSpec",
            ops_machine_ids=ops_machine_ids,
            ops_durations=ops_durations,
            ops_mask=ops_mask,
            machines_job_ids=machines_job_ids,
            machines_remaining_times=machines_remaining_times,
            action_mask=action_mask,
        )

    def action_spec(self) -> specs.MultiDiscreteArray:
        """Specifications of the action in the `JobShop` environment. The action gives each
        machine a job id ranging from 0, 1, ..., num_jobs where the last value corresponds
        to a no-op.

        Returns:
            action_spec: a `specs.MultiDiscreteArray` spec.
        """
        return specs.MultiDiscreteArray(
            num_values=jnp.full(self.num_machines, self.num_jobs + 1, jnp.int32),
            name="action",
        )

    def render(self, state: State) -> None:
        """Render the given state of the environment. This rendering shows which job (or no-op)
        is running on each machine for the current time step and previous time steps.

        Args:
            state: `State` object containing the current environment state.
        """
        self._env_viewer.render(state)

    def _observation_from_state(self, state: State) -> Observation:
        """Converts a job shop environment state to an observation.

        Args:
            state: `State` object containing the dynamics of the environment.

        Returns:
            observation: `Observation` object containing the observation of the environment.
        """

        return Observation(
            ops_machine_ids=state.ops_machine_ids,
            ops_durations=state.ops_durations,
            ops_mask=state.ops_mask,
            machines_job_ids=state.machines_job_ids,
            machines_remaining_times=state.machines_remaining_times,
            action_mask=state.action_mask,
        )

    def _is_action_valid(
        self,
        job_id: jnp.int32,
        op_id: jnp.int32,
        machine_id: jnp.int32,
        machines_job_ids: chex.Array,
        machines_remaining_times: chex.Array,
        ops_machine_ids: chex.Array,
        updated_ops_mask: chex.Array,
    ) -> Any:
        """Check whether a particular action is valid, specifically the action of scheduling
         the specified operation of the specified job on the specified machine given the
         current status of all machines.

         To achieve this, four things need to be checked:
            - The machine is available.
            - The machine is exactly the one required by the operation.
            - The job is not currently being processed on any other machine.
            - The job has not yet finished all of its operations.

        Args:
            job_id: the job in question.
            op_id: the operation of the job in question.
            machine_id: the machine in question.
            machines_job_ids: array giving which job (or no-op) each machine is working on.
            machines_remaining_times: array giving the time until available for each machine.
            ops_machine_ids: array specifying the machine needed by each operation.
            updated_ops_mask: a boolean mask indicating which operations for each job
                remain to be scheduled.

        Returns:
            Boolean representing whether the action in question is valid.
        """
        is_machine_available = machines_remaining_times[machine_id] == 0
        is_correct_machine = ops_machine_ids[job_id, op_id] == machine_id
        is_job_ready = ~jnp.any(
            (machines_job_ids == job_id) & (machines_remaining_times > 0)
        )
        is_job_finished = jnp.all(~updated_ops_mask[job_id])
        return (
            is_machine_available & is_correct_machine & is_job_ready & ~is_job_finished
        )

    def _set_busy(self, job_id: jnp.int32, action: Action) -> Any:
        """Determine, for a given action and job, whether the job is a new job to be
        scheduled.

        Args:
            job_id: identifier of a job.
            action: `Action` object whose indices and values represent the machines and
                jobs, respectively.

        Returns:
            Boolean indicating whether the specified job is a job that will be scheduled.
        """
        return jnp.any(action == job_id)

    def _create_action_mask(
        self,
        machines_job_ids: chex.Array,
        machines_remaining_times: chex.Array,
        ops_machine_ids: chex.Array,
        ops_mask: chex.Array,
    ) -> chex.Array:
        """Create the action mask based on the current status of all machines and which
        operations remain to be scheduled. Specifically, for each machine, it is checked
        whether each job (or no-op) can be scheduled on that machine. Note that we vmap
        twice: once over the jobs and once over the machines.

        Args:
            machines_job_ids: array giving which job (or no-op) each machine is working on.
            machines_remaining_times: array giving the time until available for each machine.
            ops_machine_ids: array specifying the machine needed by each operation.
            ops_mask: a boolean mask indicating which operations for each job
                remain to be scheduled.

        Returns:
            The action mask representing which jobs (or no-op) can be scheduled on each machine.
                Has shape (num_machines, num_jobs+1).
        """
        job_indexes = jnp.arange(self.num_jobs)
        machine_indexes = jnp.arange(self.num_machines)

        # Get the ID of the next operation for each job
        next_op_ids = jnp.argmax(ops_mask, axis=-1)

        # vmap over the jobs (and their ops) and vmap over the machines
        action_mask = jax.vmap(
            jax.vmap(
                self._is_action_valid, in_axes=(0, 0, None, None, None, None, None)
            ),
            in_axes=(None, None, 0, None, None, None, None),
        )(
            job_indexes,
            next_op_ids,
            machine_indexes,
            machines_job_ids,
            machines_remaining_times,
            ops_machine_ids,
            ops_mask,
        )
        no_op_mask = jnp.ones((self.num_machines, 1), bool)
        full_action_mask = jnp.concatenate([action_mask, no_op_mask], axis=-1)

        return full_action_mask
