from typing import Any

import chex
import jax.numpy as jnp
import pytest

from jumanji.jax.knapsack.env import Knapsack
from jumanji.jax.knapsack.types import Observation


class TestObservationSpec:
    env = Knapsack()
    observation_spec = env.observation_spec()
    observation = observation_spec.generate_value()

    def test_observation_spec__generate_value(self) -> None:
        """Test generating a value which conforms to the observation spec."""
        assert isinstance(self.observation, Observation)

    def test_observation_spec__validate(self) -> None:
        """Test the validation of an observation given the observation spec."""
        observation = self.observation_spec.validate(self.observation)
        # Check that a different shape breaks the validation
        with pytest.raises(ValueError):
            modified_shape_observation = observation._replace(
                problem=observation.problem[None, ...]
            )
            self.observation_spec.validate(modified_shape_observation)
        # Check that a different dtype breaks the validation
        with pytest.raises(ValueError):
            modified_dtype_observation = observation._replace(
                problem=observation.problem.astype(jnp.float16)
            )
            self.observation_spec.validate(modified_dtype_observation)
        # Check that validating another object breaks the validation
        with pytest.raises(Exception):
            self.observation_spec.validate(None)  # type: ignore

    @pytest.mark.parametrize(
        "arg_name, new_value",
        [
            ("problem_obs", observation_spec.problem_obs.replace(shape=(3, 4))),
            (
                "first_item_obs",
                observation_spec.first_item_obs.replace(name="new_name"),
            ),
            ("last_item_obs", observation_spec.last_item_obs.replace(name="new_name")),
            ("invalid_mask", observation_spec.invalid_mask.replace(shape=(3, 4))),
        ],
    )
    def test_observation_spec__replace(self, arg_name: str, new_value: Any) -> None:
        old_spec = self.observation_spec
        new_spec = old_spec.replace(**{arg_name: new_value})
        assert new_spec is not old_spec
        assert getattr(new_spec, arg_name) == new_value
        for attr_name in {
            "problem_obs",
            "first_item_obs",
            "last_item_obs",
            "invalid_mask",
        }.difference([arg_name]):
            chex.assert_equal(
                getattr(new_spec, attr_name), getattr(old_spec, attr_name)
            )
