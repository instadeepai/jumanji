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
import jax.numpy as jnp

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.packing.jobshop.instance_generator import (
    InstanceGenerator,
    RandomInstanceGenerator,
    ToyInstanceGenerator,
)
from jumanji.environments.packing.jobshop.specs import ObservationSpec
from jumanji.environments.packing.jobshop.types import State
from jumanji.types import Action, TimeStep


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

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:  # type: ignore
        pass  # type: ignore

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
