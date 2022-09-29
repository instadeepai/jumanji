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
from jumanji.environments.combinatorial.cvrp.types import Observation


class ObservationSpec(specs.Spec[Observation]):
    def __init__(
        self,
        problem_obs: specs.BoundedArray,
        position_obs: specs.DiscreteArray,
        capacity_obs: specs.BoundedArray,
        action_mask_obs: specs.BoundedArray,
    ):
        super().__init__(name="observation")
        self.problem_obs = problem_obs
        self.position_obs = position_obs
        self.capacity_obs = capacity_obs
        self.action_mask_obs = action_mask_obs

    def __repr__(self) -> str:
        return (
            "ObservationSpec(\n"
            f"\tproblem_obs={repr(self.problem_obs)},\n"
            f"\tposition_obs={repr(self.position_obs)},\n"
            f"\tcapacity_obs={repr(self.position_obs)},\n"
            f"\taction_mask_obs={repr(self.action_mask_obs)},\n"
            ")"
        )

    def generate_value(self) -> Observation:
        """Generate a value which conforms to this spec."""
        return Observation(
            problem=self.problem_obs.generate_value(),
            position=self.position_obs.generate_value(),
            capacity=self.capacity_obs.generate_value(),
            action_mask=self.action_mask_obs.generate_value(),
        )

    def validate(self, value: Observation) -> Observation:
        """Checks if a CVRP Observation conforms to the spec.

        Args:
            value: a CVRP Observation

        Returns:
            value.

        Raises:
            ValueError: if value doesn't conform to this spec.
        """
        observation = Observation(
            *jax.tree_map(
                lambda spec, v: spec.validate(v),
                (
                    self.problem_obs,
                    self.position_obs,
                    self.capacity_obs,
                    self.action_mask_obs,
                ),
                (
                    value.problem,
                    value.position,
                    value.capacity,
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
            "problem_obs": self.problem_obs,
            "position_obs": self.position_obs,
            "capacity_obs": self.capacity_obs,
            "action_mask_obs": self.action_mask_obs,
        }
        all_kwargs.update(kwargs)
        return ObservationSpec(**all_kwargs)  # type: ignore

