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
from jumanji.environments.logic.planar.types import Observation


class ObservationSpec(specs.Spec[Observation]):
    def __init__(self, nodes: specs.Array, edges: specs.Array):
        super().__init__(name=self.__class__.__name__)
        self.nodes = nodes
        self.edges = edges

    def validate(self, value: Observation) -> Observation:
        self.nodes.validate(value.nodes)
        self.edges.validate(value.edges)
        return value

    def generate_value(self) -> Observation:
        return Observation(
            nodes=self.nodes.generate_value(), edges=self.edges.generate_value()
        )

    def replace(self, **kwargs: specs.Array) -> "ObservationSpec":
        return type(self)(
            nodes=kwargs.get("nodes", self.nodes),
            edges=kwargs.get("edges", self.edges),
        )

    def __repr__(self) -> str:
        return f"{self.name}(n={self.nodes.shape}, e={self.edges.shape})"
