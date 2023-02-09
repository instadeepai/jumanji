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

from typing import Dict

import pytest
from jax import numpy as jnp

from jumanji import specs
from jumanji.environments.logic.planar.env import PlanarGraph
from jumanji.environments.logic.planar.generator import DummyGenerator
from jumanji.environments.logic.planar.reward import IntersectionCountRewardFn
from jumanji.environments.logic.planar.specs import Observation


class TestObservationSpec:
    env = PlanarGraph(
        generator=DummyGenerator(),
        reward_fn=IntersectionCountRewardFn(),
    )
    observation_spec = env.observation_spec()

    def test_observation_spec__generate_value(self) -> None:
        generate_obs = self.observation_spec.generate_value()
        assert isinstance(generate_obs, Observation)

    @pytest.mark.parametrize(
        "update",
        (
            {"nodes": specs.Array(shape=(7, 2), dtype=jnp.int32)},
            {"edges": specs.Array(shape=(5, 2), dtype=jnp.int32)},
            {
                "nodes": specs.Array(shape=(7, 2), dtype=jnp.int32),
                "edges": specs.Array(shape=(5, 2), dtype=jnp.int32),
            },
        ),
    )
    def test_observation_spec__replace(self, update: Dict[str, specs.Array]) -> None:
        observation_spec = self.env.observation_spec()
        new_spec = observation_spec.replace(**update)
        for name, spec in update.items():
            assert new_spec.__getattribute__(name).dtype == spec.dtype
            assert new_spec.__getattribute__(name).shape == spec.shape

    def test_observation_spec__validate(self) -> None:
        observation = self.observation_spec.generate_value()
        observation = self.observation_spec.validate(observation)

        # Check that a different shape breaks the validation
        with pytest.raises(ValueError):
            modified_shape_observation = observation._replace(  # type: ignore
                nodes=observation.nodes[None, ...]
            )
            self.observation_spec.validate(modified_shape_observation)

        # Check that a different dtype breaks the validation
        with pytest.raises(ValueError):
            modified_dtype_observation = observation._replace(  # type: ignore
                cube=observation.edges.astype(jnp.float16)
            )
            self.observation_spec.validate(modified_dtype_observation)

        # Check that validating another object breaks the validation
        with pytest.raises(Exception):
            self.observation_spec.validate(None)  # type: ignore
