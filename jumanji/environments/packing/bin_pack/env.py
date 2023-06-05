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
from typing import Dict, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import matplotlib
from numpy.typing import NDArray

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.packing.bin_pack.generator import Generator, RandomGenerator
from jumanji.environments.packing.bin_pack.reward import DenseReward, RewardFn
from jumanji.environments.packing.bin_pack.space import Space
from jumanji.environments.packing.bin_pack.types import (
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
from jumanji.environments.packing.bin_pack.viewer import BinPackViewer
from jumanji.tree_utils import tree_add_element, tree_slice
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


class BinPack(Environment[State]):
    """Problem of 3D bin packing, where a set of items have to be placed in a 3D container with the
    goal of maximizing its volume utilization. This environment only supports 1 bin, meaning it is
    equivalent to the 3D-knapsack problem. We use the Empty Maximal Space (EMS) formulation of this
    problem. An EMS is a 3D-rectangular space that lives inside the container and has the following
    properties:
        - It does not intersect any items, and it is not fully included into any other EMSs.
        - It is defined by 2 3D-points, hence 6 coordinates (x1, x2, y1, y2, z1, z2),
        the first point corresponding to its bottom-left location while the second defining its
        top-right corner.

    - observation: `Observation`
        - ems: `EMS` tree of jax arrays (float if `normalize_dimensions` else int32) each of
            shape (obs_num_ems,),
            coordinates of all EMSs at the current timestep.
        - ems_mask: jax array (bool) of shape (obs_num_ems,)
            indicates the EMSs that are valid.
        - items: `Item` tree of jax arrays (float if `normalize_dimensions` else int32) each of
            shape (max_num_items,),
            characteristics of all items for this instance.
        - items_mask: jax array (bool) of shape (max_num_items,)
            indicates the items that are valid.
        - items_placed: jax array (bool) of shape (max_num_items,)
            indicates the items that have been placed so far.
        - action_mask: jax array (bool) of shape (obs_num_ems, max_num_items)
            mask of the joint action space: `True` if the action (ems_id, item_id) is valid.

    - action: `MultiDiscreteArray` (int32) of shape (obs_num_ems, max_num_items).
        - ems_id: int between 0 and obs_num_ems - 1 (included).
        - item_id: int between 0 and max_num_items - 1 (included).

    - reward: jax array (float) of shape (), could be either:
        - dense: increase in volume utilization of the container due to packing the chosen item.
        - sparse: volume utilization of the container at the end of the episode.

    - episode termination:
        - if no action can be performed, i.e. no items fit in any EMSs, or all items have been
            packed.
        - if an invalid action is taken, i.e. an item that does not fit in an EMS or one that is
            already packed.

    - state: `State`
        - coordinates: jax array (float) of shape (num_nodes + 1, 2)
            the coordinates of each node and the depot.
        - demands: jax array (int32) of shape (num_nodes + 1,)
            the associated cost of each node and the depot (0.0 for the depot).
        - position: jax array (int32)
            the index of the last visited node.
        - capacity: jax array (int32)
            the current capacity of the vehicle.
        - visited_mask: jax array (bool) of shape (num_nodes + 1,)
            binary mask (False/True <--> not visited/visited).
        - trajectory: jax array (int32) of shape (2 * num_nodes,)
            identifiers of the nodes that have been visited (set to DEPOT_IDX if not filled yet).
        - num_visits: int32
            number of actions that have been taken (i.e., unique visits).

    ```python
    from jumanji.environments import BinPack
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
        generator: Optional[Generator] = None,
        obs_num_ems: int = 40,
        reward_fn: Optional[RewardFn] = None,
        normalize_dimensions: bool = True,
        debug: bool = False,
        viewer: Optional[Viewer[State]] = None,
    ):
        """Instantiates a `BinPack` environment.

        Args:
            generator: `Generator` whose `__call__` instantiates an environment
                instance. Implemented options are [`RandomGenerator`, `ToyGenerator`,
                `CSVGenerator`]. Defaults to `RandomGenerator` that generates up to 20 items maximum
                and that can handle 40 EMSs.
            obs_num_ems: number of EMSs (possible spaces in which to place an item) to show to the
                agent. If `obs_num_ems` is smaller than `generator.max_num_ems`, the first
                `obs_num_ems` largest EMSs (in terms of volume) will be returned in the observation.
                The good number heavily depends on the number of items (given by the instance
                generator). Default to 40 EMSs observable.
            reward_fn: compute the reward based on the current state, the chosen action, the next
                state, whether the transition is valid and if it is terminal. Implemented options
                are [`DenseReward`, `SparseReward`]. In each case, the total return at the end of
                an episode is the volume utilization of the container. Defaults to `DenseReward`.
            normalize_dimensions: if True, the observation is normalized (float) along each
                dimension into a unit cubic container. If False, the observation is returned in
                millimeters, i.e. integers (for both items and EMSs). Default to True.
            debug: if True, will add to timestep.extras an `invalid_ems_from_env` field that checks
                if an invalid EMS was created by the environment, which should not happen. Computing
                this metric slows down the environment. Default to False.
            viewer: `Viewer` used for rendering. Defaults to `BinPackViewer` with "human" render
                mode.
        """
        self.generator = generator or RandomGenerator(
            max_num_items=20,
            max_num_ems=40,
            split_num_same_items=2,
        )
        self.obs_num_ems = obs_num_ems
        self.reward_fn = reward_fn or DenseReward()
        self.normalize_dimensions = normalize_dimensions
        self._viewer = viewer or BinPackViewer("BinPack", render_mode="human")
        self.debug = debug

    def __repr__(self) -> str:
        return "\n".join(
            [
                "BinPack environment:",
                f" - generator: {self.generator}",
                f" - max_num_items: {self.generator.max_num_items}",
                f" - obs_num_ems: {self.obs_num_ems}",
                f" - max_num_ems: {self.generator.max_num_ems}",
                f" - reward_fn: {self.reward_fn}",
                f" - normalize_dimensions: {self.normalize_dimensions}",
                f" - debug: {self.debug}",
            ]
        )

    def observation_spec(self) -> specs.Spec[Observation]:
        """Specifications of the observation of the `BinPack` environment.

        Returns:
            Spec for the `Observation` whose fields are:
            - ems:
                - if normalize_dimensions:
                    tree of BoundedArray (float) of shape (obs_num_ems,).
                - else:
                    tree of BoundedArray (int32) of shape (obs_num_ems,).
            - ems_mask: BoundedArray (bool) of shape (obs_num_ems,).
            - items:
                - if normalize_dimensions:
                    tree of BoundedArray (float) of shape (max_num_items,).
                - else:
                    tree of BoundedArray (int32) of shape (max_num_items,).
            - items_mask: BoundedArray (bool) of shape (max_num_items,).
            - items_placed: BoundedArray (bool) of shape (max_num_items,).
            - action_mask: BoundedArray (bool) of shape (obs_num_ems, max_num_items).
        """
        obs_num_ems = self.obs_num_ems
        max_num_items = self.generator.max_num_items
        max_dim = max(self.generator.container_dims)

        if self.normalize_dimensions:
            ems_dict = {
                f"{coord_name}": specs.BoundedArray(
                    (obs_num_ems,), float, 0.0, 1.0, coord_name
                )
                for coord_name in ["x1", "x2", "y1", "y2", "z1", "z2"]
            }
        else:
            ems_dict = {
                f"{coord_name}": specs.BoundedArray(
                    (obs_num_ems,), jnp.int32, 0, max_dim, coord_name
                )
                for coord_name in ["x1", "x2", "y1", "y2", "z1", "z2"]
            }
        ems = specs.Spec(EMS, "EMSSpec", **ems_dict)
        ems_mask = specs.BoundedArray((obs_num_ems,), bool, False, True, "ems_mask")
        if self.normalize_dimensions:
            items_dict = {
                f"{axis}": specs.BoundedArray((max_num_items,), float, 0.0, 1.0, axis)
                for axis in ["x_len", "y_len", "z_len"]
            }
        else:
            items_dict = {
                f"{axis}": specs.BoundedArray(
                    (max_num_items,), jnp.int32, 0, max_dim, axis
                )
                for axis in ["x_len", "y_len", "z_len"]
            }
        items = specs.Spec(Item, "ItemsSpec", **items_dict)
        items_mask = specs.BoundedArray(
            (max_num_items,), bool, False, True, "items_mask"
        )
        items_placed = specs.BoundedArray(
            (max_num_items,), bool, False, True, "items_placed"
        )
        action_mask = specs.BoundedArray(
            (obs_num_ems, max_num_items),
            bool,
            False,
            True,
            "action_mask",
        )
        return specs.Spec(
            Observation,
            "ObservationSpec",
            ems=ems,
            ems_mask=ems_mask,
            items=items,
            items_mask=items_mask,
            items_placed=items_placed,
            action_mask=action_mask,
        )

    def action_spec(self) -> specs.MultiDiscreteArray:
        """Specifications of the action expected by the `BinPack` environment.

        Returns:
            MultiDiscreteArray (int32) of shape (obs_num_ems, max_num_items).
            - ems_id: int between 0 and obs_num_ems - 1 (included).
            - item_id: int between 0 and max_num_items - 1 (included).
        """
        num_values = jnp.array(
            [self.obs_num_ems, self.generator.max_num_items], jnp.int32
        )
        return specs.MultiDiscreteArray(num_values=num_values, name="action")

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment by calling the instance generator for a new instance.

        Args:
            key: random key used to reset the environment.

        Returns:
            state: `State` object corresponding to the new state of the environment after a reset.
            timestep: `TimeStep` object corresponding the first timestep returned by the environment
                after a reset. Also contains the following metrics in the `extras` field:
                - volume_utilization: utilization (in [0, 1]) of the container.
                - packed_items: number of items that are packed in the container.
                - ratio_packed_items: ratio (in [0, 1]) of items that are packed in the container.
                - active_ems: number of active EMSs in the current instance.
                - invalid_action: True if the action that was just taken was invalid.
                - invalid_ems_from_env (optional): True if the environment produced an EMS that was
                    invalid. Only available in debug mode.
        """
        # Generate a new instance.
        state = self.generator(key)

        # Make the observation.
        state, observation, extras = self._make_observation_and_extras(state)

        extras.update(invalid_action=jnp.array(False))
        if self.debug:
            extras.update(invalid_ems_from_env=jnp.array(False))
        timestep = restart(observation, extras)

        return state, timestep

    def step(
        self, state: State, action: chex.Array
    ) -> Tuple[State, TimeStep[Observation]]:
        """Run one timestep of the environment's dynamics. If the action is invalid, the state
        is not updated, i.e. the action is not taken, and the episode terminates.

        Args:
            state: `State` object containing the data of the current instance.
            action: jax array (int32) of shape (2,): (ems_id, item_id). This means placing the given
                item at the location of the given EMS. If the action is not valid, the flag
                `invalid_action` will be set to True in `timestep.extras` and the episode
                terminates.

        Returns:
            state: `State` object corresponding to the next state of the environment.
            timestep: `TimeStep` object corresponding to the timestep returned by the environment.
                Also contains metrics in the `extras` field:
                - volume_utilization: utilization (in [0, 1]) of the container.
                - packed_items: number of items that are packed in the container.
                - ratio_packed_items: ratio (in [0, 1]) of items that are packed in the container.
                - active_ems: number of EMSs in the current instance.
                - invalid_action: True if the action that was just taken was invalid.
                - invalid_ems_from_env (optional): True if the environment produced an EMS that was
                    invalid. Only available in debug mode.
        """
        action_is_valid = state.action_mask[tuple(action)]  # type: ignore

        obs_ems_id, item_id = action
        ems_id = state.sorted_ems_indexes[obs_ems_id]

        # Pack the item if the provided action is valid.
        next_state = jax.lax.cond(
            action_is_valid,
            lambda s: self._pack_item(s, ems_id, item_id),
            lambda s: s,
            state,
        )

        # Make the observation.
        next_state, observation, extras = self._make_observation_and_extras(next_state)

        done = ~jnp.any(next_state.action_mask) | ~action_is_valid
        reward = self.reward_fn(state, action, next_state, action_is_valid, done)

        extras.update(invalid_action=~action_is_valid)
        if self.debug:
            ems_are_all_valid = self._ems_are_all_valid(next_state)
            extras.update(invalid_ems_from_env=~ems_are_all_valid)

        timestep = jax.lax.cond(
            done,
            lambda: termination(
                reward=reward,
                observation=observation,
                extras=extras,
            ),
            lambda: transition(
                reward=reward,
                observation=observation,
                extras=extras,
            ),
        )

        return next_state, timestep

    def render(self, state: State) -> Optional[NDArray]:
        """Render the given state of the environment.

        Args:
            state: State object containing the current dynamics of the environment.
        """
        return self._viewer.render(state)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Creates an animated gif of the `BinPack` environment based on the sequence of states.

        Args:
            states: sequence of environment states corresponding to consecutive timesteps.
            interval: delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            animation.FuncAnimation: the animation object that was created.
        """
        return self._viewer.animate(states, interval, save_path)

    def close(self) -> None:
        """Perform any necessary cleanup.

        Environments will automatically :meth:`close()` themselves when
        garbage collected or when the program exits.
        """
        self._viewer.close()

    def _make_observation_and_extras(
        self, state: State
    ) -> Tuple[State, Observation, Dict]:
        """Computes the observation and the environment metrics to include in `timestep.extras`. Also
        updates the `action_mask` and `sorted_ems_indexes` in the state. The observation is obtained
        by selecting a subset of all EMSs, namely the `obs_num_ems` largest ones.
        """
        obs_ems, obs_ems_mask, sorted_ems_indexes = self._get_set_of_largest_ems(
            state.ems, state.ems_mask
        )
        state.sorted_ems_indexes = sorted_ems_indexes

        items = state.items
        action_mask = self._get_action_mask(
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

        extras = self._get_extras(state)
        return state, observation, extras

    def _get_extras(self, state: State) -> Dict:
        """Computes the environment metrics to return in `timestep.extras`.
            - volume_utilization: utilization (in [0, 1]) of the container.
            - packed_items: number of items that are packed in the container.
            - ratio_packed_items: ratio (in [0, 1]) of items that are packed in the container.
            - active_ems: number of active EMSs in the current instance.

        Args:
            state: `BinPack` state containing the current dynamics of the environment.

        Returns:
            dictionary of metrics.
        """
        items_volume = jnp.sum(item_volume(state.items) * state.items_placed)
        volume_utilization = items_volume / state.container.volume()
        packed_items = jnp.sum(state.items_placed)
        ratio_packed_items = packed_items / jnp.sum(state.items_mask)
        active_ems = jnp.sum(state.ems_mask)
        extras = {
            "volume_utilization": volume_utilization,
            "packed_items": packed_items,
            "ratio_packed_items": ratio_packed_items,
            "active_ems": active_ems,
        }
        return extras

    def _normalize_ems_and_items(
        self, state: State, obs_ems: EMS, items: Item
    ) -> Tuple[EMS, Item]:
        """Normalize the EMSs and items in the observation. Each dimension is divided by the
        container length so that they are all between 0.0 and 1.0. Hence, the ratio is not kept.
        """
        x_len, y_len, z_len = container_item = item_from_space(state.container)
        norm_space = Space(x1=x_len, x2=x_len, y1=y_len, y2=y_len, z1=z_len, z2=z_len)
        obs_ems = jax.tree_util.tree_map(
            lambda ems, container: ems / container, obs_ems, norm_space
        )
        items = jax.tree_util.tree_map(
            lambda item, container: item / container, items, container_item
        )
        return obs_ems, items

    def _ems_are_all_valid(self, state: State) -> chex.Array:
        """Checks if all EMSs are valid, i.e. they don't intersect items and do not stick out of the
        container. This check is only done in debug mode.
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

    def _get_set_of_largest_ems(
        self, ems: EMS, ems_mask: chex.Array
    ) -> Tuple[EMS, chex.Array, chex.Array]:
        """Returns a subset of EMSs by selecting the `obs_num_ems` largest EMSs."""
        ems_volumes = ems.volume() * ems_mask
        sorted_ems_indexes = jnp.argsort(
            -ems_volumes
        )  # minus sign to sort in decreasing order
        obs_ems_indexes = sorted_ems_indexes[: self.obs_num_ems]
        obs_ems = jax.tree_util.tree_map(lambda x: x[obs_ems_indexes], ems)
        obs_ems_mask = ems_mask[obs_ems_indexes]
        return obs_ems, obs_ems_mask, sorted_ems_indexes

    def _get_action_mask(
        self,
        obs_ems: EMS,
        obs_ems_mask: chex.Array,
        items: Item,
        items_mask: chex.Array,
        items_placed: chex.Array,
    ) -> chex.Array:
        """Compute the mask of valid actions.

        Args:
            obs_ems: tree of EMSs from the observation.
            obs_ems_mask: mask of EMSs.
            items: all items.
            items_mask: mask of items.
            items_placed: placing mask of items.

        Returns:
            action_mask: jax array (bool) of shape (obs_num_ems, max_num_items,).
        """

        def is_action_allowed(
            ems: EMS,
            ems_mask: chex.Array,
            item: Item,
            item_mask: chex.Array,
            item_placed: chex.Array,
        ) -> chex.Array:
            item_fits_in_ems = item_fits_in_item(item, item_from_space(ems))
            return ~item_placed & item_mask & ems_mask & item_fits_in_ems

        action_mask = jax.vmap(
            jax.vmap(is_action_allowed, in_axes=(None, None, 0, 0, 0)),
            in_axes=(0, 0, None, None, None),
        )(obs_ems, obs_ems_mask, items, items_mask, items_placed)
        return action_mask

    def _pack_item(self, state: State, ems_id: int, item_id: chex.Numeric) -> State:
        """This method assumes that the item can be placed correctly, i.e. the action is valid."""
        # Place the item in the bottom left corner of the EMS.
        ems = tree_slice(state.ems, ems_id)
        state.items_location = tree_add_element(
            state.items_location, item_id, Location(ems.x1, ems.y1, ems.z1)
        )
        state.items_placed = state.items_placed.at[item_id].set(True)
        state = self._update_ems(state, item_id)
        return state

    def _update_ems(self, state: State, item_id: chex.Numeric) -> State:
        """Update the EMSs after packing the item."""

        item_space = space_from_item_and_location(
            tree_slice(state.items, item_id), tree_slice(state.items_location, item_id)
        )

        # Delete EMSs that intersect the new item.
        ems_mask_after_intersect = ~item_space.intersect(state.ems) & state.ems_mask

        # Get the EMSs created by splitting the intersected EMSs.
        intersections_ems_dict, intersections_mask_dict = self._get_intersections_dict(
            state, item_space, ems_mask_after_intersect
        )

        # Loop over intersection EMSs from all directions to add them to the current set of EMSs.
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
        """Creates two dictionaries of intersection EMSs and intersection masks for each direction
        (lower/upper and x/y/z).
        """
        # Create new EMSs from EMSs that intersect the new item
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
        intersections_mask_dict: Dict[str, chex.Array] = jax.tree_util.tree_map(
            lambda intersections_ems: state.ems_mask
            & ~intersections_ems.is_empty()
            & ~(intersections_ems.is_included(state.ems) & ems_mask_after_intersect),
            intersections_ems_dict,
            is_leaf=lambda x: isinstance(x, Space),
        )

        # Remove intersection EMSs that are included in other intersection EMSs. The process is
        # cyclic but we can do it only once since is_included is a transitive operation.
        # This does not remove EMSs that are equal <=> both included in each other.
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

                # The alternative EMSs are included in the current direction EMSs.
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

                # Remove EMSs that are strictly included in other EMSs. This does not remove
                # EMSs that are identical and included into each other.
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
        """Add the intersection EMSs to the current set of EMSs."""

        def add_one_ems(
            carry: Tuple[EMS, chex.Array], x: Tuple[EMS, chex.Array]
        ) -> Tuple[Tuple[EMS, chex.Array], None]:
            """For loop body that goes through all EMSs to add."""
            intersection_ems, intersection_mask = x

            def add_the_ems(ems: EMS, ems_mask: chex.Array) -> Tuple[EMS, chex.Array]:
                """Function that adds the EMS conditioned on the intersection_mask value."""
                ems_index = jnp.argmin(ems_mask)
                ems = tree_add_element(ems, ems_index, intersection_ems)
                ems_mask = ems_mask.at[ems_index].set(True)
                return ems, ems_mask

            def inclusion_check(ems: EMS, ems_mask: chex.Array) -> chex.Array:
                """Only add EMSs that are not included in others."""
                is_included = intersection_ems.is_included(ems) & ems_mask
                return ~(is_included.any())

            (ems, ems_mask) = jax.lax.cond(
                intersection_mask & inclusion_check(*carry),
                add_the_ems,
                lambda *_: _,
                *carry,
            )
            return (ems, ems_mask), None

        (ems, ems_mask), _ = jax.lax.scan(
            add_one_ems, (ems, ems_mask), (intersection_ems, intersection_mask)
        )
        return ems, ems_mask
