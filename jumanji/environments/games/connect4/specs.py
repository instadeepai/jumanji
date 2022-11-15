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
from jumanji.environments.games.connect4.types import Observation


class ObservationSpec(specs.Spec[Observation]):
    def __init__(
        self,
        board_obs: specs.Array,
        action_mask: specs.Array,
        current_player: specs.DiscreteArray,
    ):
        name = (
            "Observation(\n"
            f"\tboard: {board_obs.name},\n"
            f"\taction_mask: {action_mask.name},\n"
            f"\tcurrent_player: {current_player.name},\n"
            ")"
        )
        super().__init__(name=name)
        self.board_obs = board_obs
        self.action_mask = action_mask
        self.current_player = current_player

    def __repr__(self) -> str:
        return (
            "ObservationSpec(\n"
            f"\tboard_obs={repr(self.board_obs)},\n"
            f"\taction_mask={repr(self.action_mask)},\n"
            f"\tcurrent_player={repr(self.current_player)},\n"
            ")"
        )

    def generate_value(self) -> Observation:
        """Generate values for the observations board and action_mask."""
        return Observation(
            board=self.board_obs.generate_value(),
            action_mask=self.action_mask.generate_value(),
            current_player=self.current_player.generate_value(),
        )

    def validate(self, value: Observation) -> Observation:
        """Checks if a Connect4 Observation conforms to the spec.

        Args:
            value: a Connect4 Observation

        Returns:
            value.

        Raises:
            ValueError: if value doesn't conform to this spec.
        """
        observation = Observation(
            board=self.board_obs.validate(value.board),
            action_mask=self.action_mask.validate(value.action_mask),
            current_player=self.current_player.validate(value.current_player),
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
            "board_obs": self.board_obs,
            "action_mask": self.action_mask,
            "current_player": self.current_player,
        }
        all_kwargs.update(kwargs)
        return ObservationSpec(**all_kwargs)  # type: ignore
