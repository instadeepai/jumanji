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

import chex
import jax
import pytest

from jumanji import specs
from jumanji.environments.routing.macvrp.env import MACVRP
from jumanji.environments.routing.macvrp.specs import ObservationSpec
from jumanji.environments.routing.macvrp.types import Observation


class TestObservationSpec:
    @pytest.fixture
    def observation_spec(self, macvrp_env: MACVRP) -> ObservationSpec:
        return macvrp_env.observation_spec()

    @pytest.fixture
    def observation(self, observation_spec: ObservationSpec) -> Any:
        return observation_spec.generate_value()

    def test_observation_spec__generate_value(self, macvrp_env: MACVRP) -> None:
        """Test generating a value which conforms to the observation spec."""
        assert isinstance(macvrp_env.observation_spec().generate_value(), Observation)

    def test_action_spec__generate_value(self, macvrp_env: MACVRP) -> None:
        """Test if the action spec can be called and if it is of correct
        type and size."""
        act_spec = macvrp_env.action_spec()
        assert isinstance(act_spec, specs.BoundedArray)
        assert act_spec.maximum == macvrp_env.num_customers + 1

    def test_observation_spec__validate_shape(
        self, observation_spec: ObservationSpec, observation: Observation
    ) -> None:
        """Test that a different shape of an observation element breaks the
        validation given the observation spec."""
        observation = observation_spec.validate(observation)
        modified_shape_observation = observation._replace(
            action_mask=observation.action_mask[None, ...]
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
            action_mask=observation.action_mask.astype(jax.numpy.float16)
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
                "nodes_spec",
                observation_spec._specs["nodes_spec"]
                ._specs["coordinates_spec"]
                .replace(shape=(3, 4)),
            ),
            (
                "windows_spec",
                observation_spec._specs["windows_spec"]
                ._specs["start_spec"]
                .replace(shape=(3, 4)),
            ),
            (
                "coeffs_spec",
                observation_spec._specs["coeffs_spec"]
                ._specs["early_spec"]
                .replace(shape=(3, 4)),
            ),
            (
                "other_vehicles_spec",
                observation_spec._specs["other_vehicles_spec"]
                ._specs["capacities_spec"]
                .replace(name="new_name"),
            ),
            (
                "main_vehicles_spec",
                observation_spec._specs["main_vehicles_spec"]
                ._specs["capacities_spec"]
                .replace(name="new_name"),
            ),
            (
                "action_mask_spec",
                observation_spec._specs["action_mask_spec"].replace(name="new_name"),
            ),
        ]

        for arg_name, new_value in args:
            old_spec = observation_spec
            new_spec = old_spec.replace(**{arg_name: new_value})

            assert new_spec != old_spec
            assert new_spec._specs[arg_name] == new_value

            arg_names = {
                "nodes_spec",
                "windows_spec",
                "coeffs_spec",
                "other_vehicles_spec",
                "main_vehicles_spec",
                "action_mask_spec",
            }.difference([arg_name])

            for attr_name in arg_names:
                chex.assert_equal(
                    new_spec._specs[attr_name], old_spec._specs[attr_name]
                )
