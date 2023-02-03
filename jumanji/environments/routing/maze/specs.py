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
from jumanji.environments.routing.maze.types import Observation, Position


class PositionSpec(specs.Spec[Position]):
    def __init__(
        self,
        row_spec: specs.BoundedArray,
        col_spec: specs.BoundedArray,
    ):
        super().__init__(name="position")
        self.row_spec = row_spec
        self.col_spec = col_spec

    def __repr__(self) -> str:
        return (
            "PositionSpec(\n"
            f"\trow_spec={repr(self.row_spec)},\n"
            f"\tcol_spec={repr(self.col_spec)},\n"
            ")"
        )

    def generate_value(self) -> Position:
        """Generate a value which conforms to this spec."""
        return Position(
            row=self.row_spec.generate_value(),
            col=self.col_spec.generate_value(),
        )

    def validate(self, value: Position) -> Position:
        """Checks if a Position conforms to the spec. In particular,
        it returns the Position if it conforms to the `PositionSpec` and
        raises an error otherwise.

        Args:
            value: a Position.

        Returns:
            a Maze Position.

        Raises:
            ValueError: if value doesn't conform to this spec.
        """
        position = Position(
            row=self.row_spec.validate(value.row),
            col=self.col_spec.validate(value.col),
        )
        return position

    def replace(self, **kwargs: Any) -> "PositionSpec":
        """Returns a new copy of `ObservationSpec` with specified attributes replaced.

        Args:
            **kwargs: Optional attributes to replace.

        Returns:
            A new copy of `ObservationSpec`.
        """

        all_kwargs = {
            "row_spec": self.row_spec,
            "col_spec": self.col_spec,
        }
        all_kwargs.update(kwargs)

        return PositionSpec(**all_kwargs)


class ObservationSpec(specs.Spec[Observation]):
    def __init__(
        self,
        agent_position_spec: PositionSpec,
        target_position_spec: PositionSpec,
        walls_spec: specs.BoundedArray,
        action_mask_spec: specs.BoundedArray,
        step_count_spec: specs.Array,
    ):
        super().__init__(name="observation")
        self.agent_position_spec = agent_position_spec
        self.target_position_spec = target_position_spec
        self.walls_spec = walls_spec
        self.step_count_spec = step_count_spec
        self.action_mask_spec = action_mask_spec

    def __repr__(self) -> str:
        return (
            "ObservationSpec(\n"
            f"\tagent_position_spec={repr(self.agent_position_spec)},\n"
            f"\ttarget_position_spec={repr(self.target_position_spec)},\n"
            f"\twalls_spec={repr(self.walls_spec)},\n"
            f"\tstep_count_spec={repr(self.step_count_spec)},\n"
            f"\taction_mask_spec={repr(self.action_mask_spec)},\n"
            ")"
        )

    def generate_value(self) -> Observation:
        """Generate a value which conforms to this spec."""
        return Observation(
            agent_position=self.agent_position_spec.generate_value(),
            target_position=self.target_position_spec.generate_value(),
            walls=self.walls_spec.generate_value(),
            step_count=self.step_count_spec.generate_value(),
            action_mask=self.action_mask_spec.generate_value(),
        )

    def validate(self, value: Observation) -> Observation:
        """Checks if a Maze Observation conforms to the spec. In particular,
        it returns the observation if it conforms to the `ObservationSpec` and
        raises an error otherwise.

        Args:
            value: a Maze Observation.

        Returns:
            A Maze Observation.

        Raises:
            ValueError: if value doesn't conform to this spec.
        """
        observation = Observation(
            *jax.tree_map(
                lambda spec, v: spec.validate(v),
                (
                    self.agent_position_spec,
                    self.target_position_spec,
                    self.walls_spec,
                    self.step_count_spec,
                    self.action_mask_spec,
                ),
                (
                    value.agent_position,
                    value.target_position,
                    value.walls,
                    value.step_count,
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
            "agent_position_spec": self.agent_position_spec,
            "target_position_spec": self.target_position_spec,
            "walls_spec": self.walls_spec,
            "step_count_spec": self.step_count_spec,
            "action_mask_spec": self.action_mask_spec,
        }
        all_kwargs.update(kwargs)
        return ObservationSpec(**all_kwargs)  # type: ignore
