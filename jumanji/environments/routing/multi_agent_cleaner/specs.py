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

from jumanji import specs
from jumanji.environments.routing.multi_agent_cleaner.types import Observation


class ObservationSpec(specs.Spec[Observation]):
    def __init__(
        self,
        grid_spec: specs.BoundedArray,
        agents_locations_spec: specs.BoundedArray,
        action_mask_spec: specs.BoundedArray,
    ):
        """Instantiate the specification of the Cleaner observation.

        Returns:
            - grid_spec: BoundedArray of int between 0 and 2 (inclusive), same shape as the grid.
            - agent_locations_spec: BoundedArray of int, shape is (num_agents, 2).
                Maximum value for the first column is the width of the grid,
                and maximum value for the second is the height of the grid.
            - action_mask_spec: BoundedArray of bool, shape is (num_agent, 4).
        """
        super().__init__(name="observation")
        self.grid_spec = grid_spec
        self.agents_locations_spec = agents_locations_spec
        self.action_mask_spec = action_mask_spec

    def __repr__(self) -> str:
        return (
            "ObservationSpec(\n"
            f"\tgrid_spec={self.grid_spec!r}, \n"
            f"\tagents_locations_spec={self.agents_locations_spec!r},\n"
            f"\taction_mask_spec={self.action_mask_spec!r},\n"
            ")"
        )

    def generate_value(self) -> Observation:
        """Generate a value which conforms to this spec."""
        return Observation(
            grid=self.grid_spec.generate_value(),
            agents_locations=self.agents_locations_spec.generate_value(),
            action_mask=self.action_mask_spec.generate_value(),
        )

    def validate(self, value: Observation) -> Observation:
        """Checks if an Cleaner Observation conforms to the spec.

        Args:
            value: an Cleaner Observation.

        Returns:
            value if no error is raised.

        Raises:
            ValueError: if value doesn't conform to this spec.
        """
        value = Observation(
            grid=self.grid_spec.validate(value.grid),
            agents_locations=self.agents_locations_spec.validate(
                value.agents_locations
            ),
            action_mask=self.action_mask_spec.validate(value.action_mask),
        )
        return value

    def replace(self, **kwargs: specs.BoundedArray) -> "ObservationSpec":
        """Returns a new copy of `ObservationSpec` with specified attributes replaced.

        Args:
            **kwargs: Optional attributes to replace.

        Returns:
            A new copy of `ObservationSpec`.
        """
        all_kwargs = {
            "grid_spec": self.grid_spec,
            "agents_locations_spec": self.agents_locations_spec,
            "action_mask_spec": self.action_mask_spec,
        }
        all_kwargs.update(kwargs)
        return ObservationSpec(**all_kwargs)
