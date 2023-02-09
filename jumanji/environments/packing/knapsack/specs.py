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

from jumanji import specs
from jumanji.environments.packing.knapsack.types import Observation


class ObservationSpec(specs.Spec[Observation]):
    def __init__(
        self,
        weights_spec: specs.BoundedArray,
        values_spec: specs.BoundedArray,
        action_mask_spec: specs.BoundedArray,
    ):
        super().__init__(name="observation")
        self.weights_spec = weights_spec
        self.values_spec = values_spec
        self.action_mask_spec = action_mask_spec

    def __repr__(self) -> str:
        return (
            "ObservationSpec(\n"
            f"\tweights_spec={repr(self.weights_spec)},\n"
            f"\tvalues_spec={repr(self.values_spec)},\n"
            f"\taction_mask_spec={repr(self.action_mask_spec)},\n"
            ")"
        )

    def generate_value(self) -> Observation:
        """Generate a value which conforms to this spec."""
        return Observation(
            weights=self.weights_spec.generate_value(),
            values=self.values_spec.generate_value(),
            action_mask=self.action_mask_spec.generate_value(),
        )

    def validate(self, value: Observation) -> Observation:
        """Checks if a Knapsack Observation conforms to the spec.

        Args:
            value: a Knapsack Observation.

        Returns:
            value.

        Raises:
            ValueError: if value doesn't conform to this spec.
        """
        return Observation(
            weights=self.weights_spec.validate(value.weights),
            values=self.values_spec.validate(value.values),
            action_mask=self.action_mask_spec.validate(value.action_mask),
        )

    def replace(self, **kwargs: Any) -> "ObservationSpec":
        """Returns a new copy of `ObservationSpec` with specified attributes replaced.

        Args:
            **kwargs: Optional attributes to replace.

        Returns:
            A new copy of `ObservationSpec`.
        """
        all_kwargs = {
            "weights_spec": self.weights_spec,
            "values_spec": self.values_spec,
            "action_mask_spec": self.action_mask_spec,
        }
        all_kwargs.update(kwargs)
        return ObservationSpec(**all_kwargs)  # type: ignore
