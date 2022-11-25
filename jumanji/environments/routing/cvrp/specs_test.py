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

import chex
import jax.numpy as jnp
import pytest

from jumanji.environments.routing.cvrp.env import CVRP
from jumanji.environments.routing.cvrp.specs import ObservationSpec
from jumanji.environments.routing.cvrp.types import Observation


class TestObservationSpec:
    @pytest.fixture
    def env(self) -> CVRP:
        """Instantiates a default CVRP environment."""
        return CVRP()

    @pytest.fixture
    def observation_spec(self, env: CVRP) -> ObservationSpec:
        return env.observation_spec()

    @pytest.fixture
    def observation(self, observation_spec: ObservationSpec) -> Observation:
        return observation_spec.generate_value()

    def test_observation_spec__generate_value(self, env: CVRP) -> None:
        """Test generating a value which conforms to the observation spec."""
        assert isinstance(env.observation_spec().generate_value(), Observation)

    def test_observation_spec__validate_shape(
        self, observation_spec: ObservationSpec, observation: Observation
    ) -> None:
        """Test that a different shape of an observation element breaks the
        validation given the observation spec."""
        observation = observation_spec.validate(observation)
        modified_shape_observation = observation._replace(
            coordinates=observation.coordinates[None, ...]
        )
        with pytest.raises(ValueError):
            observation_spec.validate(modified_shape_observation)

    def test_observation_spec__validate_dtype(
        self, observation_spec: ObservationSpec, observation: Observation
    ) -> None:
        """Test that a different dtype of an observation element breaks the
        validation given the observation spec."""
        observation = observation_spec.validate(observation)
        modified_dtype_observation = observation._replace(
            coordinates=observation.coordinates.astype(jnp.float16)
        )
        with pytest.raises(ValueError):
            observation_spec.validate(modified_dtype_observation)

    def test_observation_spec__validate_object(
        self, observation_spec: ObservationSpec
    ) -> None:
        """Test that validating another object breaks the validation."""
        with pytest.raises(Exception):
            observation_spec.validate(None)  # type: ignore

    def test_observation_spec__replace(self, observation_spec: ObservationSpec) -> None:
        """Test the replace method of ObservationSpec. Check that replacing the value of an
        attribute changes the observation spec and that it only changes the specific attribute
        (the remaining attributes are unchanged)."""
        args = [
            (
                "coordinates_spec",
                observation_spec.coordinates_spec.replace(shape=(3, 4)),
            ),
            ("demands_spec", observation_spec.demands_spec.replace(shape=(3, 4))),
            ("position_spec", observation_spec.position_spec.replace(name="new_name")),
            ("capacity_spec", observation_spec.capacity_spec.replace(name="new_name")),
            (
                "action_mask_spec",
                observation_spec.action_mask_spec.replace(shape=(3, 4)),
            ),
        ]

        for arg_name, new_value in args:
            old_spec = observation_spec
            new_spec = old_spec.replace(**{arg_name: new_value})
            assert new_spec != old_spec
            assert getattr(new_spec, arg_name) == new_value

            arg_names = {
                "coordinates_spec",
                "demands_spec",
                "position_spec",
                "capacity_spec",
                "action_mask_spec",
            }.difference([arg_name])

            for attr_name in arg_names:
                chex.assert_equal(
                    getattr(new_spec, attr_name), getattr(old_spec, attr_name)
                )
