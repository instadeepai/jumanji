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
import jax.numpy as jnp
import pytest

from jumanji.environments.routing.multi_agent_cleaner.env import Cleaner
from jumanji.environments.routing.multi_agent_cleaner.types import Observation


class TestObservationSpec:
    env = Cleaner(grid_width=5, grid_height=5, num_agents=3)
    observation_spec = env.observation_spec()
    observation = observation_spec.generate_value()

    def test_observation_spec__generate_value(self) -> None:
        """Test generating a value which conforms to the observation spec."""
        assert isinstance(self.observation, Observation)

    def test_observation_spec__validate_shape(self) -> None:
        """Check that a different shape breaks the validation given the observation spec."""
        obs = self.observation_spec.validate(self.observation)
        with pytest.raises(ValueError):
            modified_shape_observation = obs._replace(
                grid=obs.grid[None, ...],
                agents_locations=obs.agents_locations[None, ...],
                action_mask=obs.action_mask[None, ...],
            )
            self.observation_spec.validate(modified_shape_observation)

    def test_observation_spec__validate_dtype(self) -> None:
        """Confirm that a different dtype breaks the validation given the observation spec."""
        obs = self.observation_spec.validate(self.observation)
        with pytest.raises(ValueError):
            modified_dtype_observation = obs._replace(
                grid=obs.grid.astype(jnp.float16),
                agents_locations=obs.agents_locations.astype(jnp.float16),
                action_mask=obs.action_mask.astype(jnp.float16),
            )
            self.observation_spec.validate(modified_dtype_observation)

    def test_observation_spec__validate_object(self) -> None:
        """Confirm that validating another object breaks the validation given the observation
        spec."""
        with pytest.raises(Exception):
            self.observation_spec.validate(None)  # type: ignore

    @pytest.mark.parametrize(
        "arg_name, new_value",
        [
            (
                "grid_spec",
                observation_spec.grid_spec.replace(shape=(3, 4)),
            ),
            (
                "agents_locations_spec",
                observation_spec.agents_locations_spec.replace(name="new_name"),
            ),
            (
                "action_mask_spec",
                observation_spec.action_mask_spec.replace(shape=(3, 4)),
            ),
        ],
    )
    def test_observation_spec__replace(self, arg_name: str, new_value: Any) -> None:
        """For every attribute of `ObservationSpec`, check that we can create a new
        observation spec with a replaced attribute. Then check that the value of every
        other attribute remain equal.
        """
        old_spec = self.observation_spec
        new_spec = old_spec.replace(**{arg_name: new_value})
        assert new_spec != old_spec
        assert getattr(new_spec, arg_name) == new_value
        for attr_name in {
            "grid_spec",
            "agents_locations_spec",
            "action_mask_spec",
        }.difference([arg_name]):
            chex.assert_equal(
                getattr(new_spec, attr_name), getattr(old_spec, attr_name)
            )
