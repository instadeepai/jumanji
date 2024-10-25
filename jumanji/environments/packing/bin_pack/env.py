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

import functools
import itertools
from functools import cached_property
from typing import Any, Dict, Optional, Sequence, Tuple, cast

import chex
import jax
import jax.numpy as jnp
import matplotlib
from numpy.typing import NDArray

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.packing.bin_pack.generator import (
    ExtendedRandomGenerator,
    Generator,
    RandomGenerator,
)
from jumanji.environments.packing.bin_pack.reward import DenseReward, RewardFn
from jumanji.environments.packing.bin_pack.space import Space
from jumanji.environments.packing.bin_pack.types import (
    EMS,
    Item,
    ItemType,
    Location,
    Observation,
    State,
    ValuedItem,
    item_fits_in_item,
    item_from_space,
    item_volume,
    space_from_item_and_location,
    valued_item_from_space_and_max_value,
)
from jumanji.environments.packing.bin_pack.viewer import (
    BinPackViewer,
    ExtendedBinPackViewer,
)
from jumanji.tree_utils import tree_add_element, tree_slice
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


class BinPack(Environment[State, specs.MultiDiscreteArray, Observation]):
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
            shape (max_num_items, 6),
            characteristics of all items for this instance.
        - items_mask: jax array (bool) of shape (max_num_items, 6)
            indicates the items that are valid.
        - items_placed: jax array (bool) of shape (max_num_items, 6)
            indicates the items that have been placed so far.
        - action_mask: jax array (bool) of shape (obs_num_ems, max_num_items, 6)
            mask of the joint action space: `True` if the action (ems_id, item_id) is valid.

    - action: `MultiDiscreteArray` (int32) of shape (obs_num_ems, max_num_items, 6).
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
    key = jax.random.PRNGKey(0)
    state, timestep = jax.jit(env.reset)(key)
    env.render(state)
    action = env.action_spec.generate_value()
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
        full_support: Optional[bool] = False,
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
            full_support: if full_support is true a box can only be placed on top of a set of boxes
                only if the bottom face of the box being placed is fully supported by the top face
                of the set of supporting boxes.
        """
        self.generator = generator or RandomGenerator(
            max_num_items=20,
            max_num_ems=40,
            split_num_same_items=2,
        )
        self.obs_num_ems = obs_num_ems
        self.reward_fn = reward_fn or DenseReward()
        self.normalize_dimensions = normalize_dimensions
        super().__init__()
        self._viewer = viewer or BinPackViewer("BinPack", render_mode="human")
        self.debug = debug
        self.full_support = full_support

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

    @cached_property
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
                    tree of BoundedArray (int32) of shape (max_num_items,) though float is used for
                    values if they are valued items.
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
        items = specs.Spec(Item, "ItemsSpec", **items_dict)  # type: ignore
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

    def _items_dict_for_valued_items(self, max_num_items: int, max_dim: int) -> Dict:
        """Set the items_dict specs to the correct bounded array for valued items depending
        on whether features are to be normalized or not.

        Args:
            max_num_items: the maximum number of items that can be in an instance.
            max_dim: The maximum dimension in this given instance.

        Returns:
            A dictionary with string keys of the item features and specs BoundedArray as values.
        """
        items_dict = self._items_dict_for_non_valued_items(max_num_items, max_dim)
        if self.normalize_dimensions:
            items_dict["value"] = specs.BoundedArray(
                (self.generator.max_num_items,), float, -1.0, 1.0, "value"
            )
        else:
            items_dict["value"] = specs.BoundedArray(
                (self.generator.max_num_items,), float, -jnp.inf, jnp.inf, "value"
            )
        return items_dict

    def _items_dict_for_non_valued_items(
        self, max_num_items: int, max_dim: int
    ) -> Dict:
        """Set the items_dict specs to the correct bounded array for non valued items depending
        on whether dimensions are to be normalized or not.

        Args:
            max_num_items: the maximum number of items that can be in an instance.
            max_dim: The maximum dimension in this given instance.

        Returns:
            A dictionary with string keys of the item features and specs BoundedArray as values.
        """
        if self.normalize_dimensions:
            return {
                f"{axis}": specs.BoundedArray(
                    (self.generator.max_num_items,), float, 0.0, 1.0, axis
                )
                for axis in ["x_len", "y_len", "z_len"]
            }
        return {
            f"{axis}": specs.BoundedArray(
                (self.generator.max_num_items,), jnp.int32, 0, max_dim, axis
            )
            for axis in ["x_len", "y_len", "z_len"]
        }

    @cached_property
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
        ratio_packed_items = packed_items / state.nb_items
        active_ems = jnp.sum(state.ems_mask)
        extras = {
            "volume_utilization": volume_utilization,
            "packed_items": packed_items,
            "ratio_packed_items": ratio_packed_items,
            "active_ems": active_ems,
        }
        return extras

    def _normalize_ems_and_items(
        self, state: State, obs_ems: EMS, items: ItemType
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
        items: ItemType,
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
            item: ItemType,
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
        if self.full_support:
            self.merge_same_height_ems(state)
        return state

    def _get_intersections_dict(
        self, state: State, item_space: Space, ems_mask_after_intersect: chex.Array
    ) -> Tuple[Dict[str, Space], Dict[str, chex.Array]]:
        """Creates two dictionaries of intersection EMSs and intersection masks for each direction
        (lower/upper and x/y/z).
        """
        # Create new EMSs from EMSs that intersect the new item
        intersections_ems_dict: Dict[str, Space] = {
            f"{axis}_{direction}": item_space.hyperplane(
                axis, direction, self.full_support
            ).intersection(state.ems)
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

    def merge_same_height_ems(self, state: State) -> None:
        """
        Function that takes the state as input and merges all the EMS that are contiguous and
        have the same height into one bigger EMS.

        """
        zero_vol_ems = Space(x1=0, x2=0, y1=0, y2=0, z1=0, z2=0)
        max_nb_ems = len(state.ems.x1)

        def merge(direction: int, space1: "Space", space2: "Space") -> "Space":
            """
                Function that takes two spaces and a direction and returns a merged space from
                those two spaces along that direction.

            Args:
                direction: 1: merge along y axis, 2: merge along the x axis.
            Returns:
                Space: Space obtained from the merger of the two input spaces.
            """
            if direction == 1:
                x1 = space1.x1
                x2 = space1.x2
                y1 = jnp.minimum(space1.y1, space2.y1)
                y2 = jnp.maximum(space1.y2, space2.y2)
            elif direction == 2:
                x1 = jnp.minimum(space1.x1, space2.x1)
                x2 = jnp.maximum(space1.x2, space2.x2)
                y1 = space1.y1
                y2 = space1.y2
            return Space(x1=x1, x2=x2, y1=y1, y2=y2, z1=space1.z1, z2=space1.z2)

        def compute_merge_mask(args: Tuple) -> Tuple[chex.Array, chex.Array]:
            """Computes a boolean matrix where element i, j is True if ems i and j can be merged.

            Two EMS can be merged if they start at the same z1, either have the same width, start
            at the same y1 and they're continuous on the x axis of the container, or have the same
            length, start at the same x1 and they're continuous on the y axis of the container.
            Note, we do not verify the z2 values since ems merging in this way is only done in the
            case of a full support constraint and z2 is always equal to container.height in that
            case, ie. there are no overhanging items.

            Args:
                ems_arr: Array EMS representing all the current EMSs of the state.
                ems_mask: Mask over all the EMSs of the state.


            Returns:
                mask: 2D boolean array where mask[i,j] is True if tree_slice(EMS,i) and
                    tree_slice(EMS, j) have the same height and can be merged.
                same_x : 2D boolean array where same_x[i,j] is True if
                    tree_slice(EMS,i).x1 == tree_slice(EMS,j).x1 and
                    tree_slice(EMS,i).x2 == tree_slice(EMS,j).x2.
                    If mask[i,j]==True and same_x[i,j] == True then we can merge these EMSs along
                    the y axis of the container and if False we can merge them along the x axis of
                    the container.
            """

            def isclose_matrix(a: chex.Array, b: chex.Array) -> chex.Array:
                """
                This function takes two 1D vectors and returns a 2D boolean matrix where
                    (i,j) = True if a[i] is close to b[j].
                """
                return jnp.isclose(
                    jnp.expand_dims(a, -1) - jnp.expand_dims(b, -1).transpose(), 0
                )

            ems_arr, ems_mask = args

            same_y = isclose_matrix(ems_arr.y1, ems_arr.y1) & isclose_matrix(
                ems_arr.y2, ems_arr.y2
            )
            same_x = isclose_matrix(ems_arr.x1, ems_arr.x1) & isclose_matrix(
                ems_arr.x2, ems_arr.x2
            )
            side_by_side_x = isclose_matrix(ems_arr.x1, ems_arr.x2) | isclose_matrix(
                ems_arr.x2, ems_arr.x1
            )
            side_by_side_y = isclose_matrix(ems_arr.y1, ems_arr.y2) | isclose_matrix(
                ems_arr.y2, ems_arr.y1
            )

            # The ems have the same z1 and the emss exist.
            mask = jnp.triu(
                isclose_matrix(ems_arr.z1, ems_arr.z1)  # [nb_ems, nb_ems]
                & ems_mask  # [nb_ems, ]
            )
            # Can be merged along the y or x axis.
            mask = mask & (
                same_x & side_by_side_y | same_y & side_by_side_x
            )  # [nb_ems, nb_ems] (but only use the upper triangular part of the matrix).
            return mask, same_x

        def merge_if_possible(
            ems_arr: EMS,
            flat_can_merge_ems: chex.Array,
            same_x: chex.Array,
            is_merged_ems: chex.Array,
            mask_ind: chex.Array,
        ) -> Tuple[chex.Array, chex.Array]:
            """
               Function that merges two EMS if the merge_mask allows it.
            Args:
                ems_arr : initial array of EMSs.
                flat_can_merge_ems: 1D array obtained from flattening a 2D array returned by the
                compute_merge_mask function. flat_can_merge_ems[i] = True means that the EMSs
                i//max_nb_ems and i%max_nb_ems can be merged.
                same_x: 1D boolean array where same_x[i] is True if (
                    tree_slice(EMS,i//max_nb_ems).x1 == tree_slice(EMS,i%max_nb_ems).x1 and
                    tree_slice(EMS,i//max_nb_ems).x2 == tree_slice(EMS,i%max_nb_ems).x2).
                If flat_can_merge_ems[i]==True and same_x[i] == True then we can merge these
                EMSs along the y axis of the container and if False we can merge them along the
                x axis of the container.
                is_merged_ems: 2D triangular boolean matrix where is_merged_ems[i,j] is False
                    if the i-th EMS and the j-th EMS are already merged.
                mask_ind: the index of the element of the mask that the function will examine.
            Returns:
                Triangular boolean matrix is_merged_ems, and Space resulting from the merger of
                the EMS at  mask_ind// max_nb_ems and
                mask_ind% max_nb_ems.
            """

            # Get the indeces of the 2 emss corresponding to mask_ind in the the non-flattened
            # can_merge_ems matrix.
            row = mask_ind // max_nb_ems
            column = mask_ind % max_nb_ems
            merged_ems = jax.lax.cond(
                # make sure we can merge the two EMS located at mask_ind// max_nb_ems and
                # mask_ind% max_nb_ems, and make sure that we haven't merged the ems located at
                # row before this.
                flat_can_merge_ems[mask_ind] & ~jnp.any(is_merged_ems[row]),
                lambda _: jax.lax.cond(
                    same_x[mask_ind],
                    functools.partial(merge, 1),
                    functools.partial(merge, 2),
                    *(
                        tree_slice(ems_arr, row),
                        tree_slice(ems_arr, column),
                    ),
                ),
                lambda *_: zero_vol_ems,
                (),
            )
            is_merged_ems = is_merged_ems.at[row, column].set(~merged_ems.is_empty())
            return is_merged_ems, merged_ems

        def delete_merged_ems_and_add_new_ems(
            flat_is_merged_ems: chex.Array,
            flat_merged_ems: chex.Array,
            merged_ems_indices: Tuple[chex.Array, chex.Array],
            new_ems_and_mask: Tuple[EMS, chex.Array],
            merged_ems_ind: Tuple[chex.Array],
        ) -> Tuple[EMS, Any]:
            """
                Function that takes the a list of EMS and the indices of two merged EMS
                and removes those EMS from that list and puts in that list the EMS resulting
                from merging those two EMS.
            Args:
                flat_is_merged_ems: 1D array obtained from the flattening of the 2D array
                    is_merged_ems. flat_is_merged_ems[i] ==  True if the two ems at
                    i//max_nb_ems and i%max_nb_ems have been merged.
                flat_merged_ems is a Tree of Spaces of length max_nb_ems**2 and contains the
                    newly created EMSs from merging the initial EMS.
                    if flat_is_merged_ems[i] then flat_merged_ems[i] = EMS obtained from merging
                    the two EMSs at i//max_nb_ems and i%max_nb_ems.
                merged_ems_indices: Tuple of Arrays, where merged_ems_indices[:,merged_ems_ind]
                contains the indices in the initial ems array of the two ems that were
                merged in order to create the EMS at flat_merged_ems[
                    merged_ems_indices[0][merged_ems_ind] * max_nb_ems
                    + merged_ems_indices[1][merged_ems_ind]
                ]
                new_ems_and_mask: Tuple containing the Initial Array of EMSs and the mask
                    associated to it.
                merged_ems_ind: indices of two merged EMS.

            Returns:
                List of EMS where the merged EMS were deleted and the
                newly created EMS added.
            """
            ems_arr, ems_mask = new_ems_and_mask
            new_ems_and_mask = jax.lax.cond(
                # If the current EMS is the product of merging two EMSs together.
                flat_is_merged_ems[merged_ems_ind],
                lambda ems_arr, ems_mask: (
                    tree_add_element(
                        # Add the new EMS at the place of the first EMS used to merge.
                        tree_add_element(
                            ems_arr,
                            merged_ems_indices[0][merged_ems_ind],
                            tree_slice(
                                flat_merged_ems,
                                # The Resulting EMS is at this index because the merged_ems
                                # array is flat and has a shape of max_nb_ems ** 2
                                # (but only the first max_nb_ems make sense).
                                merged_ems_indices[0][merged_ems_ind] * max_nb_ems
                                + merged_ems_indices[1][merged_ems_ind],
                            ),
                        ),
                        # Add an empty EMS at the place of the second EMS used to merge.
                        merged_ems_indices[1][merged_ems_ind],
                        zero_vol_ems,
                    ),
                    # Set the mask to True at the index of the newly added merged EMS
                    # and to False at the index of the empty EMS.
                    ems_mask.at[merged_ems_indices[0][merged_ems_ind]]
                    .set(True)
                    .at[merged_ems_indices[1][merged_ems_ind]]
                    .set(False),
                ),
                lambda *_: _,
                ems_arr,
                ems_mask,
            )
            return new_ems_and_mask, None

        def merge_ems(args: Tuple[EMS, chex.Array]) -> Tuple[EMS, chex.Array]:
            """Function that merges all the ems it can.

            Args:
                args: Tuple containing the EMS and the EMS mask arrays.

            Returns:
                Updated EMS and EMS mask arrays after merging all the contiguous EMS having the same
                z1.
            """
            ems_arr, ems_mask = args
            # can_merge_ems = True if emss i and j can be merged.
            can_merge_ems, same_x = compute_merge_mask((ems_arr, ems_mask))

            is_merged_ems = jnp.full_like(can_merge_ems, False)

            flat_same_x = same_x.flatten()
            flat_can_merge_ems = can_merge_ems.flatten()
            # Construct new emss from merging previous ones.
            # - is_merged_ems[i,j] = True if the two ems at i and j have been merged.
            # - flat_merged_ems is a Tree of Spaces of length max_nb_ems**2 and contains the newly
            #   created EMSs from merging the initial EMS.
            is_merged_ems, flat_merged_ems = jax.lax.scan(
                functools.partial(
                    merge_if_possible, ems_arr, flat_can_merge_ems, flat_same_x
                ),
                is_merged_ems,
                jnp.arange(len(flat_can_merge_ems)),
            )
            flat_is_merged_ems = is_merged_ems.flatten()
            # Make sure that the indices of the merged EMS are at the top of this list.
            keys = flat_is_merged_ems.argsort()[::-1]
            flat_is_merged_ems = flat_is_merged_ems.sort()[::-1]
            # Create a list of tuples that are used to access the list of merged EMS.
            _, merged_ems_indices = jax.lax.scan(
                lambda _, key: (_, (key // max_nb_ems, key % max_nb_ems)),
                jnp.arange(max_nb_ems),
                keys,
            )
            # Go through the first max_nb_ems  elements of the list of merged EMS since we know that
            # it can contain at most number of initial ems given to this function.
            # Loop through the original ems tree and delete all the EMSs that were merged and add
            # the newly created EMSs.
            (new_ems, new_ems_mask), _ = jax.lax.scan(
                functools.partial(
                    delete_merged_ems_and_add_new_ems,
                    flat_is_merged_ems,
                    flat_merged_ems,
                    merged_ems_indices,
                ),
                (ems_arr, ems_mask),
                jnp.arange(max_nb_ems),
            )

            return new_ems, new_ems_mask

        state.ems, state.ems_mask = jax.lax.while_loop(
            lambda ems_and_mask: jnp.any(compute_merge_mask(ems_and_mask)[0]),
            merge_ems,
            (state.ems, state.ems_mask),
        )

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


class ExtendedBinPack(BinPack):
    def __init__(
        self,
        is_rotation_allowed: bool,
        is_value_based: bool,
        generator: Optional[Generator] = None,
        obs_num_ems: int = 40,
        reward_fn: Optional[RewardFn] = None,
        normalize_dimensions: bool = True,
        debug: bool = False,
        viewer: Optional[Viewer[State]] = None,
        mean_item_value: Optional[float] = None,
        std_item_value: Optional[float] = None,
        full_support: Optional[bool] = False,
    ):
        generator = generator or ExtendedRandomGenerator(
            is_rotation_allowed=is_rotation_allowed,
            is_value_based=is_value_based,
            max_num_items=20,
            max_num_ems=40,
            mean_item_value=mean_item_value,
            std_item_value=std_item_value,
        )
        viewer = viewer or ExtendedBinPackViewer(
            "ExtendedBinPack",
            is_rotation_allowed=is_rotation_allowed,
            render_mode="human",
        )
        super().__init__(
            generator,
            obs_num_ems,
            reward_fn,
            normalize_dimensions,
            debug,
            viewer,
            full_support,
        )
        self.is_value_based = is_value_based
        self.is_rotation_allowed = is_rotation_allowed

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

        if self.is_value_based:
            if self.is_rotation_allowed:
                items_dict = self._items_dict_for_rotated_valued_items(
                    max_num_items, max_dim
                )
            else:
                items_dict = self._items_dict_for_valued_items(max_num_items, max_dim)
        else:
            if self.is_rotation_allowed:
                items_dict = self._items_dict_for_rotated_items(max_num_items, max_dim)
            else:
                items_dict = self._items_dict_for_non_valued_items(
                    max_num_items, max_dim
                )

        items = specs.Spec(
            ValuedItem if self.is_value_based else Item, "ItemsSpec", **items_dict
        )
        nb_orientations = 1 + 5 * self.is_rotation_allowed
        items_mask = specs.BoundedArray(
            (nb_orientations * max_num_items,), bool, False, True, "items_mask"
        )
        items_placed = specs.BoundedArray(
            (nb_orientations * max_num_items,), bool, False, True, "items_placed"
        )
        action_mask = specs.BoundedArray(
            (
                obs_num_ems,
                nb_orientations * max_num_items,
            ),
            bool,
            False,
            True,
            "action_mask",
        )
        return (
            super()
            .observation_spec()
            .replace(
                items=items,
                items_mask=items_mask,
                items_placed=items_placed,
                action_mask=action_mask,
            )
        )

    def _items_dict_for_valued_items(self, max_num_items: int, max_dim: int) -> Dict:
        """Set the items_dict specs to the correct bounded array for valued items depending
        on whether features are to be normalized or not.

        Args:
            max_num_items: the maximum number of items that can be in an instance.
            max_dim: The maximum dimension in this given instance.

        Returns:
            A dictionary with string keys of the item features and specs BoundedArray as values.
        """
        items_dict = self._items_dict_for_non_valued_items(max_num_items, max_dim)
        if self.normalize_dimensions:
            items_dict["value"] = specs.BoundedArray(
                (max_num_items,), float, -1.0, 1.0, "value"
            )
        else:
            items_dict["value"] = specs.BoundedArray(
                (max_num_items,), float, -jnp.inf, jnp.inf, "value"
            )
        return items_dict

    def _items_dict_for_non_valued_items(
        self, max_num_items: int, max_dim: int
    ) -> Dict:
        """Set the items_dict specs to the correct bounded array for non valued items depending
        on whether dimensions are to be normalized or not.

        Args:
            max_num_items: the maximum number of items that can be in an instance.
            max_dim: The maximum dimension in this given instance.

        Returns:
            A dictionary with string keys of the item features and specs BoundedArray as values.
        """
        if self.normalize_dimensions:
            return {
                f"{axis}": specs.BoundedArray((max_num_items,), float, 0.0, 1.0, axis)
                for axis in ["x_len", "y_len", "z_len"]
            }
        return {
            f"{axis}": specs.BoundedArray((max_num_items,), jnp.int32, 0, max_dim, axis)
            for axis in ["x_len", "y_len", "z_len"]
        }

    def _items_dict_for_rotated_items(self, max_num_items: int, max_dim: int) -> Dict:
        """Set the items_dict specs to the correct bounded array for items depending
        on whether features are to be normalized or not.

        Args:
            max_num_items: the maximum number of items that can be in an instance.
            max_dim: The maximum dimension in this given instance.

        Returns:
            A dictionary with string keys of the item features and specs BoundedArray as values.
        """
        if self.normalize_dimensions:
            items_dict = {
                f"{axis}": specs.BoundedArray(
                    (6 * max_num_items,), float, 0.0, 1.0, axis
                )
                for axis in ["x_len", "y_len", "z_len"]
            }

        else:
            items_dict = {
                f"{axis}": specs.BoundedArray(
                    (6 * max_num_items,), jnp.int32, 0, max_dim, axis
                )
                for axis in ["x_len", "y_len", "z_len"]
            }
        return items_dict

    def _items_dict_for_rotated_valued_items(
        self, max_num_items: int, max_dim: int
    ) -> Dict:
        """Set the items_dict specs to the correct bounded array for valued items depending
        on whether features are to be normalized or not.

        Args:
            max_num_items: the maximum number of items that can be in an instance.
            max_dim: The maximum dimension in this given instance.

        Returns:
            A dictionary with string keys of the item features and specs BoundedArray as values.
        """
        items_dict = self._items_dict_for_rotated_items(max_num_items, max_dim)
        if self.normalize_dimensions:
            items_dict["value"] = specs.BoundedArray(
                (6 * max_num_items,), float, -1.0, 1.0, "value"
            )
        else:
            items_dict["value"] = specs.BoundedArray(
                (6 * max_num_items,), float, -jnp.inf, jnp.inf, "value"
            )

        return items_dict

    def action_spec(self) -> specs.MultiDiscreteArray:
        """Specifications of the action expected by the `BinPack` environment.

        Returns:
            MultiDiscreteArray (int32) of shape (obs_num_ems, max_num_items).
            - ems_id: int between 0 and obs_num_ems - 1 (included).
            - item_id: int between 0 and max_num_items - 1 (included).
        """
        if self.is_rotation_allowed:
            num_values = jnp.array(
                [6, self.obs_num_ems, self.generator.max_num_items], jnp.int32
            )
        else:
            num_values = jnp.array(
                [self.obs_num_ems, self.generator.max_num_items], jnp.int32
            )
        return specs.MultiDiscreteArray(num_values=num_values, name="action")

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
        orientation, obs_ems_id, item_id = None, None, None
        if self.is_rotation_allowed:
            orientation, obs_ems_id, item_id = action
        else:
            obs_ems_id, item_id = action
        ems_id = state.sorted_ems_indexes[obs_ems_id]

        # Pack the item if the provided action is valid.
        next_state = jax.lax.cond(
            action_is_valid,
            lambda s: self._pack_item(s, ems_id, item_id, orientation),
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

    def _make_observation_and_extras(
        self, state: State
    ) -> Tuple[State, Observation, Dict]:
        """Computes the observation and the environment metrics to include in `timestep.extras`.
        Also updates the `action_mask` and `sorted_ems_indexes` in the state. The observation is
        obtained by selecting a subset of all EMSs, namely the `obs_num_ems` largest ones.

        Args:
            state: a state of the ExtendedBinPack environment.

        Returns:
            - State with the updated EMSs and EMS mask.
            - Observation with the valid action_mask, item_placed and items_mask.
            - extra metrics that can be used to measure the performance of the agent.

        """

        def flatten_observation(observation: Observation) -> Observation:
            """In the case where item rotation is allowed, this function is used to

            Args:
                observation: Initial observation with items, items_placed and items_mask arrays of
                shape (6, max_nb_items) and an action mask of shape (6, max_nb_ems, max_nb_items)

            Returns:
                Observation where the items, items_placed and items_mask array have a shape
                (6 * max_nb_items) and the action_mask has a shape of (max_nb_ems, 6 * max_nb_items)
            """
            flattened_items_mask = observation.items_mask.flatten()
            flattened_items_placed = observation.items_placed.flatten()
            flattened_action_mask = observation.action_mask.reshape(
                observation.action_mask.shape[1],
                -1,
            )
            if self.is_value_based:
                items = cast(
                    ValuedItem,
                    observation.items,
                )
                return Observation(
                    ems=observation.ems,
                    ems_mask=observation.ems_mask,
                    items=ValuedItem(
                        items.x_len.flatten(),
                        items.y_len.flatten(),
                        items.z_len.flatten(),
                        items.value.flatten(),
                    ),
                    items_mask=flattened_items_mask,
                    items_placed=flattened_items_placed,
                    action_mask=flattened_action_mask,
                )

            else:
                return Observation(
                    ems=observation.ems,
                    ems_mask=observation.ems_mask,
                    items=Item(
                        observation.items.x_len.flatten(),
                        observation.items.y_len.flatten(),
                        observation.items.z_len.flatten(),
                    ),
                    items_mask=flattened_items_mask,
                    items_placed=flattened_items_placed,
                    action_mask=flattened_action_mask,
                )

        state, observation, extra = super()._make_observation_and_extras(state)
        flat_obs = observation
        if self.is_rotation_allowed:
            flat_obs = flatten_observation(flat_obs)
        return state, flat_obs, extra

    def _normalize_ems_and_items(
        self, state: State, obs_ems: EMS, items: ItemType  # type: ignore
    ) -> Tuple[EMS, Item]:
        """Normalize the EMSs and items in the observation. Each dimension is divided by the
        container length so that they are all between 0.0 and 1.0. Hence, the ratio is not kept.
        """
        # If items have the extra feature: value (for cases where we want to maximize the value
        # packed into a container instead of the volume) we normalise by the largest valued item
        # (observed better performances than normalising with respect to the total value of all
        # items).
        container_item: ItemType
        if isinstance(items, ValuedItem):
            items = cast(ValuedItem, items)
            state.items = cast(ValuedItem, state.items)
            (
                x_len,
                y_len,
                z_len,
                _,
            ) = container_item = valued_item_from_space_and_max_value(
                state.container, state.instance_max_item_value_magnitude
            )
        else:
            items = cast(Item, items)
            x_len, y_len, z_len = container_item = item_from_space(state.container)

        norm_space = Space(x1=x_len, x2=x_len, y1=y_len, y2=y_len, z1=z_len, z2=z_len)
        obs_ems = jax.tree_util.tree_map(
            lambda ems, normalization_ems: ems / normalization_ems,
            obs_ems,
            norm_space,
        )
        items = jax.tree_util.tree_map(
            lambda item, normalization_items: item / normalization_items,
            items,
            container_item,
        )
        return obs_ems, items

    def _pack_item(  # type: ignore
        self,
        state: State,
        ems_id: int,
        item_id: chex.Numeric,
        item_orientation: Optional[int] = None,
    ) -> State:
        """This method assumes that the item can be placed correctly, i.e. the action is valid."""
        # Place the item in the bottom left corner of the EMS.
        ems = tree_slice(state.ems, ems_id)
        state.items_location = tree_add_element(
            state.items_location, item_id, Location(ems.x1, ems.y1, ems.z1)
        )
        if item_orientation is not None:
            state.items_mask = state.items_mask.at[:, item_id].set(False)
            state.items_placed = state.items_placed.at[item_orientation, item_id].set(
                True
            )
        else:
            state.items_mask = state.items_mask.at[item_id].set(False)
            state.items_placed = state.items_placed.at[item_id].set(True)

        state = self._update_ems(state, item_id, item_orientation)
        return state

    def _get_action_mask(
        self,
        obs_ems: EMS,
        obs_ems_mask: chex.Array,
        items: ItemType,
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

        if self.is_rotation_allowed:
            expanded_obs_state = jax.tree_util.tree_map(
                functools.partial(jnp.expand_dims, axis=0), obs_ems
            )
            expanded_obs_ems_mask = jax.tree_util.tree_map(
                functools.partial(jnp.expand_dims, axis=0), obs_ems_mask
            )
            action_mask = jax.vmap(
                jax.vmap(is_action_allowed, in_axes=(None, None, 1, 1, 1)),
                in_axes=(1, 1, None, None, None),
            )(
                expanded_obs_state,
                expanded_obs_ems_mask,
                items,
                items_mask,
                items_placed,
            )
            action_mask = jnp.moveaxis(action_mask, -1, 0)
            return jnp.asarray(action_mask, bool)
        else:
            return super()._get_action_mask(
                obs_ems, obs_ems_mask, items, items_mask, items_placed
            )

    def _ems_are_all_valid(self, state: State) -> chex.Array:
        """Checks if all EMSs are valid, i.e. they don't intersect items and do not stick out of the
        container. This check is only done in debug mode.
        """
        ems_intersection_with_items = jnp.zeros((state.ems_mask.shape), bool)
        if self.is_rotation_allowed:
            for o in range(6):
                tmp_items = Item(
                    state.items[:][0][o], state.items[:][1][o], state.items[:][2][o]
                )
                item_spaces = space_from_item_and_location(
                    tmp_items, state.items_location
                )
                ems_intersect_items = jax.vmap(Space.intersect, in_axes=(0, None))(
                    state.ems, item_spaces
                )
                ems_intersect_items &= jnp.outer(state.ems_mask, state.items_placed[o])
                ems_intersection_with_items |= jnp.any(ems_intersect_items)
            ems_outside_container = jnp.any(
                state.ems_mask & ~state.ems.is_included(state.container)
            )
            return ~ems_intersection_with_items & ~ems_outside_container
        else:
            return super()._ems_are_all_valid(state)

    def _update_ems(  # type: ignore
        self, state: State, item_id: chex.Numeric, item_orientation
    ) -> State:
        """Update the EMSs after packing the item."""
        if item_orientation is not None:
            item_space = space_from_item_and_location(
                tree_slice(tree_slice(state.items, item_orientation), item_id),
                tree_slice(state.items_location, item_id),
            )
        else:
            item_space = space_from_item_and_location(
                tree_slice(state.items, item_id),
                tree_slice(state.items_location, item_id),
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
        if self.full_support:
            self.merge_same_height_ems(state)

        return state
