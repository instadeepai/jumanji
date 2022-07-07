import jax

from jumanji.jax import specs
from jumanji.jax.knapsack.types import Observation


class ObservationSpec(specs.Spec[Observation]):
    def __init__(
        self,
        problem_obs: specs.BoundedArray,
        first_item_obs: specs.DiscreteArray,
        last_item_obs: specs.DiscreteArray,
        invalid_mask: specs.BoundedArray,
    ):
        super().__init__(name="observation")
        self.problem_obs = problem_obs
        self.first_item_obs = first_item_obs
        self.last_item_obs = last_item_obs
        self.invalid_mask = invalid_mask

    def __repr__(self) -> str:
        return (
            "ObservationSpec(\n"
            f"\tproblem_obs={repr(self.problem_obs)},\n"
            f"\tfirst_item_obs={repr(self.first_item_obs)},\n"
            f"\tlast_item_obs={repr(self.last_item_obs)},\n"
            f"\tinvalid_mask={repr(self.invalid_mask)},\n"
            ")"
        )

    def generate_value(self) -> Observation:
        """Generate a value which conforms to this spec."""
        return Observation(
            problem=self.problem_obs.generate_value(),
            first_item=self.first_item_obs.generate_value(),
            last_item=self.last_item_obs.generate_value(),
            invalid_mask=self.invalid_mask.generate_value(),
        )

    def validate(self, value: Observation) -> Observation:
        """Checks if a Knapsack Observation conforms to the spec.

        Args:
            value: a Knapsack Observation

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
                    self.first_item_obs,
                    self.last_item_obs,
                    self.invalid_mask,
                ),
                (value.problem, value.first_item, value.last_item, value.invalid_mask),
            )
        )
        return observation
