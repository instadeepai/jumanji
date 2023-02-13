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
from jumanji.environments.routing.connector.types import Observation


class ObservationSpec(specs.Spec[Observation]):
    def __init__(
        self,
        grid_spec: specs.BoundedArray,
        action_mask_spec: specs.BoundedArray,
        step_spec: specs.BoundedArray,
    ):
        super().__init__(name="observation")
        self.grid_spec = grid_spec
        self.action_mask_spec = action_mask_spec
        self.step_spec = step_spec

    def __repr__(self) -> str:
        return (
            "ObservationSpec(\n"
            f"\tgrid_spec={repr(self.grid_spec)},\n"
            f"\taction_mask_spec={repr(self.action_mask_spec)},\n"
            f"\tstep_spec={repr(self.step_spec)},\n"
            ")"
        )

    def generate_value(self) -> Observation:
        """Generate a value which conforms to this spec."""
        return Observation(
            grid=self.grid_spec.generate_value(),
            action_mask=self.action_mask_spec.generate_value(),
            step=self.step_spec.generate_value(),
        )

    def validate(self, value: Observation) -> Observation:
        """Checks if a Connector Observation conforms to the spec.

        Args:
            value: a Connector Observation.

        Returns:
            value.

        Raises:
            ValueError: if value doesn't conform to this spec.
        """
        observation = Observation(
            *jax.tree_map(
                lambda spec, v: spec.validate(v),
                (self.grid_spec, self.action_mask_spec, self.step_spec),
                (value.grid, value.action_mask, value.step),
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
            "grid_spec": self.grid_spec,
            "action_mask_spec": self.action_mask_spec,
            "step_spec": self.step_spec,
        }
        all_kwargs.update(kwargs)

        return ObservationSpec(**all_kwargs)  # type: ignore
