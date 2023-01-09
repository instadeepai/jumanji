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
from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.packing.jobshop.instance_generator import (
    InstanceGenerator,
    RandomInstanceGenerator,
    ToyInstanceGenerator,
)
from jumanji.environments.packing.jobshop.specs import ObservationSpec
from jumanji.environments.packing.jobshop.types import Observation, State
from jumanji.types import Action, TimeStep, restart


class JobShop(Environment[State]):
    def __init__(
        self, instance_generator_type: str = "toy", **instance_generator_kwargs: Any
    ):
        """Instantiate a JobShop environment.

        Args:
            instance_generator_type: string representing the `InstanceGenerator` responsible for
                resetting the environment. E.g. can be a random generator to learn generalisation
                or one that outputs a hardcoded instance. Defaults to "toy" which creates the
                `ToyInstanceGenerator` that always resets to the same hardcoded instance.
                Possible values: 'toy' (default) or 'random'.
            instance_generator_kwargs: keyword arguments for the specified instance generator.
        """
        self.instance_generator = self._create_instance_generator(
            instance_generator_type, **instance_generator_kwargs
        )
        self.num_jobs = self.instance_generator.num_jobs
        self.num_machines = self.instance_generator.num_machines
        self.max_num_ops = self.instance_generator.max_num_ops
        self.max_op_duration = self.instance_generator.max_op_duration

    def __repr__(self) -> str:
        return "\n".join(
            [
                "JobShop environment:",
                f" - instance_generator: {self.instance_generator}",
                f" - num_jobs: {self.num_jobs}",
                f" - num_machines: {self.num_machines}",
                f" - max_num_ops: {self.max_num_ops}",
                f" - max_op_duration: {self.max_op_duration}",
            ]
        )

    @classmethod
    def _create_instance_generator(
        cls, instance_generator_type: str, **instance_generator_kwargs: Any
    ) -> InstanceGenerator:
        """
        Factory method for creating an instance generator.

        This method can be overridden to add new instance generator types.

        Args:
            instance_generator_type: the type of instance generator to create. Possible values:
                - 'toy': create a toy instance generator.
                - 'random': create a random instance generator.
            **instance_generator_kwargs: additional keyword arguments to pass to the instance
                generator constructor.

        Returns:
            An instance of `InstanceGenerator`.

        Raises:
            ValueError: If an unexpected value is provided for `instance_generator_type`.
        """
        instance_generator_obj: InstanceGenerator

        if instance_generator_type == "toy":
            instance_generator_obj = ToyInstanceGenerator()
        elif instance_generator_type == "random":
            instance_generator_obj = RandomInstanceGenerator(
                **instance_generator_kwargs
            )
        else:
            raise ValueError(
                f"Unexpected value for 'instance_generator_type', got {instance_generator_type!r}."
                "Possible values: 'toy', 'random'."
            )
        return instance_generator_obj

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Resets the environment by calling the instance generator for a new instance.

        Args:
            key: random key used to reset the environment.

        Returns:
            state: `State` object corresponding to the new state of the environment after the reset.
            timestep: `TimeStep` object corresponding the first timestep returned by the environment
                after a reset.
        """
        # Generate a new instance
        state = self.instance_generator(key)

        # Create the action mask and update the state
        state.action_mask = self._create_action_mask(
            state.machines_job_ids,
            state.machines_remaining_times,
            state.operations_machine_ids,
            state.operations_mask,
        )

        # Get the observation and the timestep
        obs = self._observation_from_state(state)
        timestep = restart(observation=obs)

        return state, timestep

    def step(self, state: State, action: Action) -> Tuple[State, TimeStep]:  # type: ignore
        pass  # type: ignore

    def observation_spec(self) -> ObservationSpec:
        """Specifications of the observation of the `JobShop` environment.

        Returns:
            `ObservationSpec` containing all the specifications for all the `Observation` fields:
            - operations_machine_ids: Array (jnp.int32) of shape (num_jobs, max_num_ops).
            - operations_durations: Array (jnp.int32) of shape (num_jobs, max_num_ops).
            - operations_mask: Array (bool) of shape (num_jobs, max_num_ops).
            - machines_job_ids: Array (jnp.int32) of shape (num_machines,).
            - machines_remaining_times: Array (jnp.int32) of shape (num_machines,).
            - action_mask: Array (bool) of shape (num_machines, num_jobs + 1).
        """
        operations_machine_ids = specs.BoundedArray(
            shape=(self.num_jobs, self.max_num_ops),
            dtype=jnp.int32,
            minimum=-1,
            maximum=self.num_machines - 1,
            name="operations_machine_ids",
        )
        operations_durations = specs.BoundedArray(
            shape=(self.num_jobs, self.max_num_ops),
            dtype=jnp.int32,
            minimum=-1,
            maximum=self.max_op_duration,
            name="operations_durations",
        )
        operations_mask = specs.BoundedArray(
            shape=(self.num_machines, self.max_num_ops),
            dtype=bool,
            minimum=False,
            maximum=True,
            name="operations_mask",
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
        return ObservationSpec(
            operations_machine_ids=operations_machine_ids,
            operations_durations=operations_durations,
            operations_mask=operations_mask,
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

    @staticmethod
    def _observation_from_state(state: State) -> Observation:
        """Converts a job shop environment state to an observation.

        Args:
            state: `State` object containing the dynamics of the environment.

        Returns:
            observation: `Observation` object containing the observation of the environment.
        """

        return Observation(
            operations_machine_ids=state.operations_machine_ids,
            operations_durations=state.operations_durations,
            operations_mask=state.operations_mask,
            machines_job_ids=state.machines_job_ids,
            machines_remaining_times=state.machines_remaining_times,
            action_mask=state.action_mask,
        )

    @staticmethod
    def _is_action_valid(
        job_id: jnp.int32,
        op_id: jnp.int32,
        machine_id: jnp.int32,
        machines_job_ids: chex.Array,
        machines_remaining_times: chex.Array,
        operations_machine_ids: chex.Array,
        updated_operations_mask: chex.Array,
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
            operations_machine_ids: array specifying the machine needed by each operation.
            updated_operations_mask: a boolean mask indicating which operations for each job
                remain to be scheduled.

        Returns:
            Boolean representing whether the action in question is valid.
        """
        is_machine_available = machines_remaining_times[machine_id] == 0
        is_correct_machine = operations_machine_ids[job_id, op_id] == machine_id
        is_job_ready = ~jnp.any(
            (machines_job_ids == job_id) & (machines_remaining_times > 0)
        )
        is_job_finished = jnp.all(~updated_operations_mask[job_id])
        return (
            is_machine_available & is_correct_machine & is_job_ready & ~is_job_finished
        )

    def _create_action_mask(
        self,
        machines_job_ids: chex.Array,
        machines_remaining_times: chex.Array,
        operations_machine_ids: chex.Array,
        operations_mask: chex.Array,
    ) -> chex.Array:
        """Create the action mask based on the current status of all machines and which
        operations remain to be scheduled. Specifically, for each machine, it is checked
        whether each job (or no-op) can be scheduled on that machine. Note that we vmap
        twice: once over the jobs and once over the machines.

        Args:
            machines_job_ids: array giving which job (or no-op) each machine is working on.
            machines_remaining_times: array giving the time until available for each machine.
            operations_machine_ids: array specifying the machine needed by each operation.
            operations_mask: a boolean mask indicating which operations for each job
                remain to be scheduled.

        Returns:
            The action mask representing which jobs (or no-op) can be scheduled on each machine.
                Has shape (num_machines, num_jobs+1).
        """
        job_indexes = jnp.arange(self.num_jobs)
        machine_indexes = jnp.arange(self.num_machines)

        # Get the ID of the next operation for each job
        next_op_ids = jnp.argmax(operations_mask, axis=-1)

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
            operations_machine_ids,
            operations_mask,
        )
        no_op_mask = jnp.ones((self.num_machines, 1), bool)
        full_action_mask = jnp.concatenate([action_mask, no_op_mask], axis=-1)

        return full_action_mask
