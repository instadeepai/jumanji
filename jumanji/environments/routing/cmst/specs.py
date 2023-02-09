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
from jumanji.environments.routing.cmst.types import Observation


class ObservationSpec(specs.Spec[Observation]):
    def __init__(
        self,
        node_types_spec: specs.BoundedArray,
        edges_spec: specs.BoundedArray,
        position_spec: specs.BoundedArray,
        action_mask_spec: specs.BoundedArray,
    ):
        super().__init__(name="observation")
        self.node_types_spec = node_types_spec
        self.edges_spec = edges_spec
        self.position_spec = position_spec
        self.action_mask_spec = action_mask_spec

    def __repr__(self) -> str:
        return (
            "ObservationSpec(\n"
            f"\tnode_types_spec={repr(self.node_types_spec)},\n"
            f"\tedges_spec={repr(self.edges_spec)},\n"
            f"\tposition_spec={repr(self.position_spec)},\n"
            f"\taction_mask_spec={repr(self.action_mask_spec)},\n"
            ")"
        )

    def generate_value(self) -> Observation:
        """Generate a value which conforms to this spec."""
        return Observation(
            node_types=self.node_types_spec.generate_value(),
            edges=self.edges_spec.generate_value(),
            position=self.position_spec.generate_value(),
            action_mask=self.action_mask_spec.generate_value(),
        )

    def validate(self, value: Observation) -> Observation:
        """Checks if a CMST Observation conforms to the spec.

        Args:
            value: a CMST Observation.

        Returns:
            value.

        Raises:
            ValueError: if value doesn't conform to this spec.
        """
        observation = Observation(
            *jax.tree_map(
                lambda spec, v: spec.validate(v),
                (
                    self.node_types_spec,
                    self.edges_spec,
                    self.position_spec,
                    self.action_mask_spec,
                ),
                (
                    value.node_types,
                    value.edges,
                    value.position,
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
            "node_types_spec": self.node_types_spec,
            "edges_spec": self.edges_spec,
            "position_spec": self.position_spec,
            "action_mask_spec": self.action_mask_spec,
        }
        all_kwargs.update(kwargs)

        return ObservationSpec(**all_kwargs)  # type: ignore
