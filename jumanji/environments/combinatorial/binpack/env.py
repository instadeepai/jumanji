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

import itertools
from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
from chex import PRNGKey

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.combinatorial.binpack import env_viewer
from jumanji.environments.combinatorial.binpack.instance_generator import (
    CSVInstanceGenerator,
    InstanceGenerator,
    RandomInstanceGenerator,
    ToyInstanceGenerator,
)
from jumanji.environments.combinatorial.binpack.reward import (
    RewardFn,
    sparse_linear_reward,
)
from jumanji.environments.combinatorial.binpack.space import Space
from jumanji.environments.combinatorial.binpack.specs import (
    EMSSpec,
    ItemSpec,
    ObservationSpec,
)
from jumanji.environments.combinatorial.binpack.types import (
    EMS,
    Item,
    Location,
    Observation,
    State,
    item_fits_in_item,
    item_from_space,
    item_volume,
    space_from_item_and_location,
)
from jumanji.tree_utils import tree_add_element, tree_slice
from jumanji.types import Action, TimeStep, restart, termination, transition


class BinPack(Environment[State]):
    """
    RL Environment for the problem of bin packing. We use the Empty Maximal Space (EMS) formulation
    of this problem. An EMS is a 3D-rectangular space that lives inside the container and has the
    following properties:
        - It does not intersect any items, and it is not fully included into any other EMS.
        - It is defined by 2 3D-points, hence 6 coordinates (x1, x2, y1, y2, z1, z2),
        the first point corresponding to its bottom-left location while the second defining its
        top-right corner.

    The environment has the following characteristics.

    - observation: Observation
        - ems: EMS dataclass of jax arrays (float if normalize_dimensions else int)
            of shape (obs_num_ems,).
            (x1, x2, y1, y2, z1, z2): the coordinates of all ems at the current timestep.
        - ems_mask: jax array (bool) of shape (obs_num_ems,).
            True if the ems exists.
        - items: Item dataclass of jax arrays (float if normalize_dimensions else int)
            of shape (max_num_items,).
            (x_len, y_len, z_len): characteristics of all items for this instance.
        - items_mask: jax array (bool) of shape (max_num_items,).
            True if the item exists.
        - items_placed: jax array (bool) of shape (max_num_items,).
            True if the item is placed in the container.
        - action_mask: jax array (bool) of shape (obs_num_ems, max_num_items).
            Mask of the joint action space: True if the action (ems_id, item_id) is feasible.

    - action: BoundedArray
        - ems_id: int between 0 and obs_num_ems - 1 (included).
        - item_id: int between 0 and max_num_items - 1 (included).

    - reward: jax array (float).
        Default (sparse_linear): volume utilization of the container at the end of the episode.

    - episode termination:
        if no other actions are possible (no items fit in any ems).

    ```python
    env = BinPack()
    key = jax.random.key(0)
    state, timestep = jax.jit(env.reset)(key)
    env.render(state)
    action = env.action_spec().generate_value()
    state, timestep = jax.jit(env.step)(state, action)
    env.render(state)
    ```
    """

    def __init__(
        self,
        instance_generator_type: str = "toy",
        obs_num_ems: int = 60,
        reward_fn: RewardFn = sparse_linear_reward,
        normalize_dimensions: bool = True,
        debug: bool = False,
        **instance_generator_kwargs: Any,
    ):
        """Instantiate a BinPack environment.

        Args:
            instance_generator_type: string representing the InstanceGenerator responsible for
                resetting the environment. E.g. can be a random generator to learn generalisation
                or one that outputs the same instance to do active search on that instance.
                Defaults to "toy" which creates a ToyInstanceGenerator that always resets to
                the same instance with 20 items.
                Possible values: 'toy' (default), 'csv' or 'random'.
            obs_num_ems: number of ems to show to the agent. If `obs_num_ems` is smaller than
                `generator.max_num_ems`, the first `obs_num_ems` biggest ems will be returned
                in the observation. Default to 60, but the good number heavily depends on the
                number of items (given by the instance generator).
            reward_fn: function of the state, the next state, the action taken and the done flag,
                and outputs a scalar, namely the reward. Default: sparse_linear, meaning a
                sparse reward is given at the end of the episode, corresponding to the volume
                utilization of the container.
            normalize_dimensions: if True (default), the observation is normalized to a unit cubic
                container. If False, the observation is returned in integers (both items and EMS).
            debug: if True, will output an `invalid_ems_from_env` field in the extras returned
                within timestep. Default to False as computing this metric slows down the
                environment.
            instance_generator_kwargs: Keyword arguments for the specified instance generator.
        """
        self.instance_generator = self.create_instance_generator(
            instance_generator_type, **instance_generator_kwargs
        )

        self.obs_num_ems = obs_num_ems
        self.reward_fn = reward_fn
        self.normalize_dimensions = normalize_dimensions
        self._env_viewer = env_viewer.BinPackViewer("BinPack")
        self.debug = debug

    def __repr__(self) -> str:
        return "\n".join(
            [
                "BinPack environment:",
                f" - instance_generator_type: {self.instance_generator}",
                f" - max_num_items: {self.instance_generator.max_num_items}",
                f" - obs_num_ems: {self.obs_num_ems}",
                f" - max_num_ems: {self.instance_generator.max_num_ems}",
                f" - reward_fn: {self.reward_fn}",
                f" - normalize_dimensions: {self.normalize_dimensions}",
                f" - debug: {self.debug}",
            ]
        )

    @classmethod
    def create_instance_generator(
        cls, instance_generator_type: str, **instance_generator_kwargs: Any
    ) -> InstanceGenerator:
        """
        Factory method for creating an instance generator.

        This method can be overridden to add new instance generator types.

        Args:
            instance_generator_type: The type of instance generator to create. Possible values:
                - 'toy': Create a toy instance generator.
                - 'csv': Create a CSV instance generator.
                - 'random': Create a random instance generator.
            **instance_generator_kwargs:
                Additional keyword arguments to pass to the instance generator constructor.

        Returns:
            An instance of `InstanceGenerator`.

        Raises:
            ValueError: If an unexpected value is provided for `instance_generator_type`.
        """
        instance_generator_obj: InstanceGenerator

        if instance_generator_type == "toy":
            instance_generator_obj = ToyInstanceGenerator()
        elif instance_generator_type == "csv":
            instance_generator_obj = CSVInstanceGenerator(**instance_generator_kwargs)
        elif instance_generator_type == "random":
            instance_generator_obj = RandomInstanceGenerator(
                **instance_generator_kwargs
            )
        else:
            raise ValueError(
                f"Unexpected value for 'instance_generator_type', got {instance_generator_type!r}."
                "Possible values: 'toy', 'csv', 'random'."
            )
        return instance_generator_obj

    def observation_spec(self) -> ObservationSpec:
        """Specifications of the observation of the BinPack environment.

        Returns:
            ObservationSpec containing all the specifications for all the Observation fields:
            - ems_spec: EMSSpec.
                - normalize_dimensions: True -> tree of BoundedArray (float)
                    of shape (obs_num_ems,).
                - normalize_dimensions: False -> tree of BoundedArray (int)
                    of shape (obs_num_ems,).
            - ems_mask_spec: BoundedArray (bool) of shape (obs_num_ems,).
            - items_spec: ItemSpec.
                - normalize_dimensions: True -> tree of BoundedArray (float)
                    of shape (max_num_items,).
                - normalize_dimensions: False -> tree of BoundedArray (int)
                    of shape (max_num_items,).
            - items_mask_spec: BoundedArray (bool) of shape (max_num_items,).
            - items_placed_spec: BoundedArray (bool) of shape (max_num_items,).
            - action_mask_spec: BoundedArray (bool) of shape (obs_num_ems, max_num_items).
        """
        obs_num_ems = self.obs_num_ems
        max_num_items = self.instance_generator.max_num_items
        max_dim = max(self.instance_generator.container_dims)

        if self.normalize_dimensions:
            ems_spec_dict = {
                f"{coord_name}_spec": specs.BoundedArray(
                    (obs_num_ems,), float, 0.0, 1.0, coord_name
                )
                for coord_name in ["x1", "x2", "y1", "y2", "z1", "z2"]
            }
        else:
            ems_spec_dict = {
                f"{coord_name}_spec": specs.BoundedArray(
                    (obs_num_ems,), jnp.int32, 0, max_dim, coord_name
                )
                for coord_name in ["x1", "x2", "y1", "y2", "z1", "z2"]
            }
        ems_spec = EMSSpec(**ems_spec_dict)
        ems_mask_spec = specs.BoundedArray(
            (obs_num_ems,), bool, False, True, "ems_mask"
        )
        if self.normalize_dimensions:
            items_spec_dict = {
                f"{axis}_spec": specs.BoundedArray(
                    (max_num_items,), float, 0.0, 1.0, axis
                )
                for axis in ["x_len", "y_len", "z_len"]
            }
        else:
            items_spec_dict = {
                f"{axis}_spec": specs.BoundedArray(
                    (max_num_items,), jnp.int32, 0, max_dim, axis
                )
                for axis in ["x_len", "y_len", "z_len"]
            }
        items_spec = ItemSpec(**items_spec_dict)
        items_mask_spec = specs.BoundedArray(
            (max_num_items,), bool, False, True, "items_mask"
        )
        items_placed_spec = specs.BoundedArray(
            (max_num_items,), bool, False, True, "items_placed"
        )
        action_mask_spec = specs.BoundedArray(
            (obs_num_ems, max_num_items),
            bool,
            False,
            True,
            "action_mask",
        )
        return ObservationSpec(
            ems_spec=ems_spec,
            ems_mask_spec=ems_mask_spec,
            items_spec=items_spec,
            items_mask_spec=items_mask_spec,
            items_placed_spec=items_placed_spec,
            action_mask_spec=action_mask_spec,
        )

    def action_spec(self) -> specs.MultiDiscreteArray:
        """Specifications of the action expected by the BinPack environment.

        Returns:
            MultiDiscreteArray (int) of shape (obs_num_ems, max_num_items).
            - ems_id: int between 0 and obs_num_ems - 1 (included).
            - item_id: int between 0 and max_num_items - 1 (included).
        """
        return specs.MultiDiscreteArray(
            num_values=jnp.array(
                [self.obs_num_ems, self.instance_generator.max_num_items], int
            ),
            name="action",
        )

    def reset(self, key: PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment by calling the instance generator for a new instance.

        Args:
            key: random key used to reset the environment.

        Returns:
            state: State object corresponding to the new state of the environment after a reset.
            timestep: TimeStep object corresponding the first timestep returned by the environment
                after a reset. Also contains metrics in the `extras` field:
                - volume_utilization: utilization (in [0, 1]) of the container.
                - packed_items: number of items that are packed in the container.
                - ratio_packed_items: ratio (in [0, 1]) of items that are packed in the container.
                - active_ems: number of EMS in the current instance.
                - invalid_action_taken: True if the action that was just taken was invalid.
                - invalid_ems_from_env (optional): True if the environment produced an EMS that was
                    invalid. Only available in debug mode.
        """
        # Generate a new instance.
        state = self.instance_generator(key)

        # Make the observation.
        state, observation, extras = self._make_observation_and_metrics(state)

        extras.update(invalid_action_taken=jnp.array(False))
        if self.debug:
            extras.update(invalid_ems_from_env=jnp.array(False))
        timestep = restart(observation=observation, extras=extras)

        return state, timestep

    def step(self, state: State, action: Action) -> Tuple[State, TimeStep[Observation]]:
        """Run one timestep of the environment's dynamics. If the action is invalid, the state
        is not updated, i.e. the action is not taken, and the episode terminates.

        Args:
            state: State object containing the data of the current instance.
            action: jax array of shape (2,) (ems_id, item_id). This means placing the given item
                at the location of the given ems. If the action is not feasible, the flag
                `invalid_action_taken` will be set to True in `timestep.extras` and the episode
                terminates.

        Returns:
            state: State object corresponding to the next state of the environment.
            timestep: TimeStep object corresponding to the timestep returned by the environment.
                Also contains metrics in the `extras` field:
                - volume_utilization: utilization (in [0, 1]) of the container.
                - packed_items: number of items that are packed in the container.
                - ratio_packed_items: ratio (in [0, 1]) of items that are packed in the container.
                - active_ems: number of EMS in the current instance.
                - invalid_action_taken: True if the action that was just taken was invalid.
                - invalid_ems_from_env (optional): True if the environment produced an EMS that was
                    invalid. Only available in debug mode.
        """
        chex.assert_shape(action, (2,))
        action_is_valid = state.action_mask[tuple(action)]  # type: ignore

        obs_ems_id, item_id = action
        ems_id = state.sorted_ems_indexes[obs_ems_id]

        # Pack the item is the provided action is valid.
        next_state = jax.lax.cond(
            action_is_valid,
            lambda state: self._pack_item(state, ems_id, item_id),
            lambda state: state,
            state,
        )

        # Make the observation.
        next_state, observation, extras = self._make_observation_and_metrics(next_state)

        done = ~jnp.any(next_state.action_mask) | ~action_is_valid
        reward = self.reward_fn(state, next_state, action, done)

        extras.update(invalid_action_taken=~action_is_valid)
        if self.debug:
            ems_are_valid = self._ems_are_valid(next_state)
            extras.update(invalid_ems_from_env=~ems_are_valid)

        timestep: TimeStep[Observation] = jax.lax.cond(
            done,
            lambda reward, observation, extras: termination(
                reward=reward,
                observation=observation,
                extras=extras,
            ),
            lambda reward, observation, extras: transition(
                reward=reward,
                observation=observation,
                extras=extras,
            ),
            reward,
            observation,
            extras,
        )

        return next_state, timestep

    def render(self, state: State) -> None:
        """Render the given state of the environment.

        Args:
            state: State object containing the current dynamics of the environment.
        """
        self._env_viewer.render(state)

    def close(self) -> None:
        """Perform any necessary cleanup.

        Environments will automatically :meth:`close()` themselves when
        garbage collected or when the program exits.
        """
        self._env_viewer.close()

    def _make_observation_and_metrics(
        self, state: State
    ) -> Tuple[State, Observation, Dict]:
        """Computes the observation and the environment metrics. Also updates the `action_mask` and
        `sorted_ems_indexes` in the state. The observation is obtained by selecting a subset of
        all EMS, namely the `obs_num_ems` largest ones.
        """
        obs_ems, obs_ems_mask, sorted_ems_indexes = self._subset_of_ems(
            state.ems, state.ems_mask
        )
        state.sorted_ems_indexes = sorted_ems_indexes

        items = state.items
        action_mask = self._action_mask(
            obs_ems,
            obs_ems_mask,
            items,
            state.items_mask,
            state.items_placed,
        )
        state.action_mask = action_mask

        if self.normalize_dimensions:
            obs_ems, items = self._normalize_ems_and_items(state, obs_ems, items)
        observation = Observation(
            ems=obs_ems,
            ems_mask=obs_ems_mask,
            items=items,
            items_mask=state.items_mask,
            items_placed=state.items_placed,
            action_mask=action_mask,
        )

        metrics = self.get_metrics(state)
        return state, observation, metrics

    def get_metrics(self, state: State) -> Dict:
        """Computes environment metrics.
            - volume_utilization: utilization (in [0, 1]) of the container.
            - packed_items: number of items that are packed in the container.
            - ratio_packed_items: ratio (in [0, 1]) of items that are packed in the container.
            - active_ems: number of EMS in the current instance.

        Args:
            state: BinPack state containing the current dynamics of the environment.

        Returns:
            dictionary of metrics.
        """
        items_volume = jnp.sum(item_volume(state.items) * state.items_placed)
        container_volume = state.container.volume()
        volume_utilization = items_volume / container_volume
        packed_items = jnp.sum(state.items_placed)
        nb_items = jnp.sum(state.items_mask)
        ratio_packed_items = packed_items / nb_items
        active_ems = jnp.sum(state.ems_mask)
        metrics = {
            "volume_utilization": volume_utilization,
            "packed_items": packed_items,
            "ratio_packed_items": ratio_packed_items,
            "active_ems": active_ems,
        }
        return metrics

    def _normalize_ems_and_items(
        self, state: State, obs_ems: EMS, items: Item
    ) -> Tuple[EMS, Item]:
        """Normalize the EMS and items in the observation. Each dimension is divided by the
        container length so that they are all between 0.0 and 1.0. Hence, the ratio is not kept.
        """
        container_space_to_normalize = Space(
            x1=state.container.x2,
            x2=state.container.x2,
            y1=state.container.y2,
            y2=state.container.y2,
            z1=state.container.z2,
            z2=state.container.z2,
        )
        obs_ems = jax.tree_map(
            lambda x, c: x / c, obs_ems, container_space_to_normalize
        )
        items = jax.tree_map(
            lambda x, c: x / c, items, item_from_space(state.container)
        )
        return obs_ems, items

    def _ems_are_valid(self, state: State) -> jnp.bool_:
        """Checks if all EMS are valid, i.e. they don't intersect items and do not stick out of the
        container.
        """
        item_spaces = space_from_item_and_location(state.items, state.items_location)
        ems_intersect_items = jax.vmap(Space.intersect, in_axes=(0, None))(
            state.ems, item_spaces
        )
        ems_intersect_items &= jnp.outer(state.ems_mask, state.items_placed)
        ems_intersection_with_items = jnp.any(ems_intersect_items)
        ems_outside_container = jnp.any(
            state.ems_mask & ~state.ems.is_included(state.container)
        )
        return ~ems_intersection_with_items & ~ems_outside_container

    def _subset_of_ems(
        self, ems: EMS, ems_mask: chex.Array
    ) -> Tuple[EMS, chex.Array, chex.Array]:
        """Returns a subset of EMS by selecting the `obs_num_ems` biggest EMS."""
        ems_volumes = ems.volume() * ems_mask
        sorted_ems_indexes = jnp.flip(jnp.argsort(ems_volumes))
        obs_ems = jax.tree_map(lambda x: x[sorted_ems_indexes][: self.obs_num_ems], ems)
        obs_ems_mask = ems_mask[sorted_ems_indexes][: self.obs_num_ems]
        return obs_ems, obs_ems_mask, sorted_ems_indexes

    def _action_mask(
        self,
        obs_ems: EMS,
        obs_ems_mask: chex.Array,
        items: Item,
        items_mask: chex.Array,
        items_placed: chex.Array,
    ) -> chex.Array:
        """Compute the mask of legal actions.

        Args:
            obs_ems: tree of ems from the observation.
            obs_ems_mask: mask of ems.
            items: all items.
            items_mask: mask of items.
            items_placed: placing mask of items.

        Returns:
            action_mask: jax array (bool) of shape (obs_num_ems, max_num_items,).
        """

        def is_action_allowed(
            ems: EMS,
            ems_mask: jnp.bool_,
            item: Item,
            item_mask: jnp.bool_,
            item_placed: jnp.bool_,
        ) -> jnp.bool_:
            item_fits_in_ems = item_fits_in_item(item, item_from_space(ems))
            return ~item_placed & item_mask & ems_mask & item_fits_in_ems

        action_mask = jax.vmap(
            jax.vmap(is_action_allowed, in_axes=(None, None, 0, 0, 0)),
            in_axes=(0, 0, None, None, None),
        )(obs_ems, obs_ems_mask, items, items_mask, items_placed)
        return action_mask

    def _pack_item(
        self, state: State, ems_id: chex.Numeric, item_id: chex.Numeric
    ) -> State:
        """This method assumes that the item can be placed correctly, i.e. the action is valid."""
        # Place the item in the bottom left corner of the EMS.
        state.items_location = Location(
            x=state.items_location.x.at[item_id].set(state.ems.x1[ems_id]),
            y=state.items_location.y.at[item_id].set(state.ems.y1[ems_id]),
            z=state.items_location.z.at[item_id].set(state.ems.z1[ems_id]),
        )
        state.items_placed = state.items_placed.at[item_id].set(True)

        state = self._update_ems(state, item_id)
        return state

    def _update_ems(self, state: State, item_id: chex.Numeric) -> State:
        """Update the EMS after packing the item."""

        item_space = space_from_item_and_location(
            tree_slice(state.items, item_id), tree_slice(state.items_location, item_id)
        )

        # Delete ems that intersect the new item.
        ems_mask_after_intersect = ~item_space.intersect(state.ems) & state.ems_mask

        # Get the EMS created by splitting the intersected EMS.
        intersections_ems_dict, intersections_mask_dict = self._get_intersections_dict(
            state, item_space, ems_mask_after_intersect
        )

        # Loop over intersection EMS from all directions to add them to the current set of EMS.
        new_ems = state.ems
        new_ems_mask = ems_mask_after_intersect
        for intersection_ems, intersection_mask in zip(
            intersections_ems_dict.values(), intersections_mask_dict.values()
        ):
            new_ems, new_ems_mask = self._add_ems(
                intersection_ems, intersection_mask, new_ems, new_ems_mask
            )

        state.ems = new_ems
        state.ems_mask = new_ems_mask
        return state

    def _get_intersections_dict(
        self, state: State, item_space: Space, ems_mask_after_intersect: chex.Array
    ) -> Tuple[Dict[str, Space], Dict[str, chex.Array]]:
        """Creates two dictionaries of intersection EMS and intersection masks for each direction
        (lower/upper and x/y/z).
        """
        # Create new ems from ems that intersect the new item
        intersections_ems_dict: Dict[str, Space] = {
            f"{axis}_{direction}": item_space.hyperplane(axis, direction).intersection(
                state.ems
            )
            for axis, direction in itertools.product(
                ["x", "y", "z"], ["lower", "upper"]
            )
        }

        # A new EMS is added if the intersection is not empty and if it is not fully included in
        # the EMS that do not intersect with the item.
        intersections_mask_dict: Dict[str, chex.Array] = jax.tree_map(
            lambda intersections_ems: state.ems_mask
            & ~intersections_ems.is_empty()
            & ~(intersections_ems.is_included(state.ems) & ems_mask_after_intersect),
            intersections_ems_dict,
            is_leaf=lambda x: isinstance(x, Space),
        )

        # Remove intersection EMS that are included in other intersection EMS. The process is cyclic
        # but we can do it only once since is_included is a transitive operation.
        # This does not remove EMS that are equal <=> both included in each other.
        num_ems = len(state.ems_mask)
        # Outer loop iterates through directions.
        for (direction, direction_intersections_ems), (
            _,
            direction_intersections_mask,
        ) in zip(intersections_ems_dict.items(), intersections_mask_dict.items()):

            # Inner loop iterates through alternative directions.
            for (alt_direction, alt_direction_intersections_ems), (
                _,
                alt_direction_intersections_mask,
            ) in zip(intersections_ems_dict.items(), intersections_mask_dict.items()):

                # The current direction EMS is included in the alternative EMS.
                directions_included_in_alt_directions = jax.vmap(
                    jax.vmap(Space.is_included, in_axes=(None, 0)), in_axes=(0, None)
                )(direction_intersections_ems, alt_direction_intersections_ems)
                if direction == alt_direction:
                    directions_included_in_alt_directions = (
                        directions_included_in_alt_directions.at[
                            jnp.arange(num_ems), jnp.arange(num_ems)
                        ].set(False)
                    )
                directions_included_in_alt_directions = (
                    directions_included_in_alt_directions
                    & jnp.outer(
                        direction_intersections_mask, alt_direction_intersections_mask
                    )
                )

                # The alternative EMS are included in the current direction EMS.
                alt_directions_included_in_directions = jax.vmap(
                    jax.vmap(Space.is_included, in_axes=(None, 0)), in_axes=(0, None)
                )(alt_direction_intersections_ems, direction_intersections_ems)
                if direction == alt_direction:
                    alt_directions_included_in_directions = (
                        alt_directions_included_in_directions.at[
                            jnp.arange(num_ems), jnp.arange(num_ems)
                        ].set(False)
                    )
                alt_directions_included_in_directions = (
                    alt_directions_included_in_directions
                    & jnp.outer(
                        alt_direction_intersections_mask, direction_intersections_mask
                    )
                )

                # Remove EMS that are strictly included in another EMS. This does not remove
                # EMS that are identical and included into each other.
                to_remove = jnp.any(
                    directions_included_in_alt_directions
                    & ~alt_directions_included_in_directions.T,
                    axis=-1,
                )

                intersections_mask_dict[direction] &= ~to_remove
        return intersections_ems_dict, intersections_mask_dict

    def _add_ems(
        self,
        intersection_ems: EMS,
        intersection_mask: chex.Array,
        ems: EMS,
        ems_mask: chex.Array,
    ) -> Tuple[EMS, chex.Array]:
        """Add the intersection EMS to the current set of EMS."""

        def add_one_ems(
            carry: Tuple[EMS, chex.Array], x: Tuple[EMS, chex.Array]
        ) -> Tuple[Tuple[EMS, chex.Array], None]:
            """For loop body that goes through all EMS to add."""
            intersection_ems, intersection_mask = x

            def add_the_ems(carry: Tuple[EMS, chex.Array]) -> Tuple[EMS, chex.Array]:
                """Function that adds the EMS conditioned on the intersection_mask value."""
                ems, ems_mask = carry
                i_ems = jnp.argmin(ems_mask)
                ems = tree_add_element(ems, i_ems, intersection_ems)
                ems_mask = ems_mask.at[i_ems].set(True)
                return ems, ems_mask

            def inclusion_check(carry: Tuple[EMS, chex.Array]) -> jnp.bool_:
                """Only add EMS that are not included in others."""
                ems, ems_mask = carry
                is_included = intersection_ems.is_included(ems) & ems_mask
                return ~(is_included.any())

            carry = jax.lax.cond(
                intersection_mask & inclusion_check(carry),
                add_the_ems,
                lambda _: _,
                carry,
            )
            return carry, None

        init_carry = ems, ems_mask
        (ems, ems_mask), _ = jax.lax.scan(
            add_one_ems, init_carry, (intersection_ems, intersection_mask)
        )
        return ems, ems_mask
