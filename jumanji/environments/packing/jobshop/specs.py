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

from typing import Any

import jax

from jumanji import specs
from jumanji.environments.packing.jobshop.types import Observation


class ObservationSpec(specs.Spec[Observation]):
    def __init__(
        self,
        operations_machine_ids: specs.BoundedArray,
        operations_durations: specs.BoundedArray,
        operations_mask: specs.BoundedArray,
        machines_job_ids: specs.BoundedArray,
        machines_remaining_times: specs.BoundedArray,
        action_mask: specs.BoundedArray,
    ):
        super().__init__(name="observation")
        self.operations_machine_ids = operations_machine_ids
        self.operations_durations = operations_durations
        self.operations_mask = operations_mask
        self.machines_job_ids = machines_job_ids
        self.machines_remaining_times = machines_remaining_times
        self.action_mask = action_mask

    def __repr__(self) -> str:
        return (
            "ObservationSpec(\n"
            f"\toperations_machine_ids={repr(self.operations_machine_ids)},\n"
            f"\toperations_durations={repr(self.operations_durations)},\n"
            f"\toperations_mask={repr(self.operations_mask)},\n"
            f"\tmachines_job_ids={repr(self.machines_job_ids)},\n"
            f"\tmachines_remaining_times={repr(self.machines_remaining_times)},\n"
            f"\taction_mask={repr(self.action_mask)},\n"
            ")"
        )

    def generate_value(self) -> Observation:
        """Generate a value which conforms to this spec."""
        return Observation(
            operations_machine_ids=self.operations_machine_ids.generate_value(),
            operations_durations=self.operations_durations.generate_value(),
            operations_mask=self.operations_mask.generate_value(),
            machines_job_ids=self.machines_job_ids.generate_value(),
            machines_remaining_times=self.machines_remaining_times.generate_value(),
            action_mask=self.action_mask.generate_value(),
        )

    def validate(self, value: Observation) -> Observation:
        """Checks if a JobShop Observation conforms to the spec. In particular,
        it returns the observation if it conforms to the `ObservationSpec` and
        raises an error otherwise.

        Args:
            value: a JobShop Observation.

        Returns:
            A JobShop Observation.

        Raises:
            ValueError: if value doesn't conform to this spec.
        """
        observation = Observation(
            *jax.tree_map(
                lambda spec, v: spec.validate(v),
                (
                    self.operations_machine_ids,
                    self.operations_durations,
                    self.operations_mask,
                    self.machines_job_ids,
                    self.machines_remaining_times,
                    self.action_mask,
                ),
                (
                    value.operations_machine_ids,
                    value.operations_durations,
                    value.operations_mask,
                    value.machines_job_ids,
                    value.machines_remaining_times,
                    value.action_mask,
                ),
            )
        )
        return observation

    def replace(self, **kwargs: Any) -> "ObservationSpec":
        """Returns a new copy of `ObservationSpec` with specified attributes replaced.

        Args:
            **kwargs: Optional attributes to replace.

        Returns:
            A new copy of `ObservationSpec`.
        """
        all_kwargs = {
            "operations_machine_ids": self.operations_machine_ids,
            "operations_durations": self.operations_durations,
            "operations_mask": self.operations_mask,
            "machines_job_ids": self.machines_job_ids,
            "machines_remaining_times": self.machines_remaining_times,
            "action_mask": self.action_mask,
        }
        all_kwargs.update(kwargs)
        return ObservationSpec(**all_kwargs)
