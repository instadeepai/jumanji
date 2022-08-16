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

from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp
from chex import PRNGKey

from jumanji import specs
from jumanji.binpack.instance_generator import InstanceGenerator
from jumanji.binpack.reward import sparse_linear_reward
from jumanji.binpack.specs import EMSSpec, ItemSpec, ObservationSpec
from jumanji.binpack.types import EMS, Item, Observation, RewardFn, State
from jumanji.env import Environment
from jumanji.types import Action, Extra, TimeStep, restart, termination, transition


class BinPack(Environment[State]):
    """
    RL Environment for the problem of bin packing. We use the Empty Maximal Space (EMS) formulation
    of this problem. An EMS is a 3D-rectangle space that lives inside the container and has the
    following properties: it does not intersect any items and it is not fully included into any
    other EMS. It is defined by 2 3D-points, hence 6 coordinates (x1, x2, y1, y2, z1, z2),
    the first point corresponding to its bottom-left location while the second defining its
    top-right corner.
    The environment has the following characteristics.

    - observation: Observation
        - ems: EMS dataclass of jax arrays (float) of shape (obs_num_ems,).
            (x1, x2, y1, y2, z1, z2): the coordinates of all ems at the current timestep.
        - ems_mask: jax array (bool) of shape (obs_num_ems,).
            True if the ems exists.
        - items: Item dataclass of jax arrays (float) of shape (max_num_items,).
            (x_len, y_len, z_len): characteristics of all items for this instance.
        - items_mask: jax array (bool) of shape (max_num_items,).
            True if the item exists.
        - items_placed: jax array (bool) of shape (max_num_items,).
            True if the item is placed in the container.
        - action_mask: jax array (bool) of shape (obs_num_ems, max_num_items).
            Mask of the joint action space: True if the action (ems_id, item_id) is feasible.

    - reward: jax array (float).
        Default (sparse_linear): negative of the proportion of remaining space in the container at
        the end of an episode.

    - episode termination:
        if no other actions are possible (no items fit in any ems).

    ```python
    instance_generator = SimpleInstanceGenerator(max_num_ems=40)
    env = BinPack(instance_generator, obs_num_ems=20)
    key = jax.random.key(0)
    state, timestep, _ = jax.jit(env.reset)(key)
    env.render(state)
    action = env.action_spec().generate_value()
    state, timestep, _ = jax.jit(env.step)(state, action)
    env.render(state)
    ```
    """

    def __init__(
        self,
        instance_generator: InstanceGenerator,
        obs_num_ems: int,
        reward_fn: RewardFn = sparse_linear_reward,
    ):
        """Instantiate a BinPack environment.

        Args:
            instance_generator: InstanceGenerator responsible for resetting the environment. E.g.
                can be a random generator to learn generalisation or one that outputs the same
                instance to do active search on that instance. It must inherit from the
                InstanceGenerator abstract class.
            obs_num_ems: number of ems to show to the agent. If `obs_num_ems` is smaller than
                `generator.max_num_ems`, the first `obs_num_ems` biggest ems will be returned
                in the observation.
            reward_fn: function of the state, the action taken and the done flag, and
                outputs a scalar, namely the reward. Default: sparse_linear, meaning a
                sparse reward is given at the end of the episode, corresponding to the negative
                of the proportion of remaining space inside the container.
        """
        self.instance_generator = instance_generator
        self.obs_num_ems = obs_num_ems
        self.reward_fn = reward_fn

    def __repr__(self) -> str:
        return "\n".join(
            [
                "BinPack environment:",
                f" - instance_generator: {self.instance_generator}",
                f" - max_num_items: {self.instance_generator.max_num_items}",
                f" - obs_num_ems: {self.obs_num_ems}",
                f" - max_num_ems: {self.instance_generator.max_num_ems}",
            ]
        )

    def reset(self, key: PRNGKey) -> Tuple[State, TimeStep[Observation], Extra]:
        """Resets the environment by calling the instance generator for a new instance.

        Args:
            key: random key used to reset the environment.

        Returns:
            state: State object corresponding to the new state of the environment after a reset.
            timestep: TimeStep object corresponding the first timestep returned by the environment
                after a reset.
            extra: None.

        """
        state = self.instance_generator(key)

        # Compute timestep
        # TODO: select subset of EMS to return in the observation
        obs_ems = state.ems
        obs_ems_mask = state.ems_mask

        action_mask = self._action_mask(
            obs_ems,
            obs_ems_mask,
            state.items,
            state.items_mask,
            state.items_placed,
        )
        state.action_mask = action_mask
        observation = Observation(
            ems=obs_ems,
            ems_mask=obs_ems_mask,
            items=state.items,
            items_mask=state.items_mask,
            items_placed=state.items_placed,
            action_mask=action_mask,
        )
        timestep = restart(observation=observation)
        # TODO: add metrics to extras
        extras = None
        return state, timestep, extras

    def step(
        self, state: State, action: Action
    ) -> Tuple[State, TimeStep[Observation], Extra]:
        """Run one timestep of the environment's dynamics.

        Args:
            state: State object containing the data of the current instance.
            action: jax array of shape (2,): (ems_id, item_id). This means placing the given item
                at the location of the given ems. The action is supposed to be feasible, otherwise
                the behaviour is unexpected and the flag `invalid_action_taken` will be True in
                extra.

        Returns:
            state: State object corresponding to the next state of the environment.
            timestep: TimeStep object corresponding to the timestep returned by the environment.
            extra: None.
        """

        # TODO: Actually do the env step logic
        chex.assert_shape(action, (2,))
        sorted_ems_id, item_id = action

        # Compute timestep
        # TODO: select subset of EMS to return in the observation
        obs_ems = state.ems
        obs_ems_mask = state.ems_mask

        action_mask = self._action_mask(
            obs_ems,
            obs_ems_mask,
            state.items,
            state.items_mask,
            state.items_placed,
        )
        state.action_mask = action_mask
        observation = Observation(
            ems=obs_ems,
            ems_mask=obs_ems_mask,
            items=state.items,
            items_mask=state.items_mask,
            items_placed=state.items_placed,
            action_mask=action_mask,
        )
        done = ~jnp.any(action_mask)
        timestep: TimeStep[Observation] = jax.lax.cond(
            done,
            lambda _: termination(
                reward=self.reward_fn(state, action, done), observation=observation
            ),
            lambda _: transition(
                reward=self.reward_fn(state, action, done), observation=observation
            ),
            None,
        )
        # TODO: add metrics to extras
        extras = None
        return state, timestep, extras

    def observation_spec(self) -> ObservationSpec:
        """Specifications of the observation of the BinPack environment.

        Returns:
            ObservationSpec containing all the specifications for all the Observation fields:
            - ems_spec: EMSSpec.
                - tree of Array (float) of shape (obs_num_ems,).
            - ems_mask_spec: BoundedArray (bool) of shape (obs_num_ems,).
            - items_spec: ItemSpec.
                - tree of BoundedArray (float) of shape (max_num_items,).
            - items_mask_spec: BoundedArray (bool) of shape (max_num_items,).
            - items_placed_spec: BoundedArray (bool) of shape (max_num_items,).
            - action_mask_spec: BoundedArray (bool) of shape (obs_num_ems, max_num_items).
        """
        obs_num_ems = self.obs_num_ems
        max_num_items = self.instance_generator.max_num_items

        ems_spec_dict = {
            coord_name + "_spec": specs.Array((obs_num_ems,), jnp.float32, coord_name)
            for coord_name in ["x1", "x2", "y1", "y2", "z1", "z2"]
        }
        ems_spec = EMSSpec(**ems_spec_dict)
        ems_mask_spec = specs.BoundedArray(
            (obs_num_ems,), jnp.bool_, False, True, "ems_mask"
        )
        items_spec_dict = {
            axis_len
            + "_spec": specs.BoundedArray(
                (max_num_items,), jnp.float32, 0.0, jnp.inf, axis_len
            )
            for axis_len in ["x_len", "y_len", "z_len"]
        }
        items_spec = ItemSpec(**items_spec_dict)
        items_mask_spec = specs.BoundedArray(
            (max_num_items,), jnp.bool_, False, True, "items_mask"
        )
        items_placed_spec = specs.BoundedArray(
            (max_num_items,), jnp.bool_, False, True, "items_placed"
        )
        action_mask_spec = specs.BoundedArray(
            (obs_num_ems, max_num_items),
            jnp.bool_,
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

    def action_spec(self) -> specs.BoundedArray:
        """Specifications of the action expected by the BinPack environment.

        Returns:
            BoundedArray (int) of shape (2,).
            - ems_id: int between 0 and obs_num_ems - 1 (included).
            - item_id: int between 0 and max_num_items - 1 (included).
        """
        return specs.BoundedArray(
            shape=(2,),
            dtype=jnp.int32,
            minimum=(0, 0),
            maximum=(self.obs_num_ems - 1, self.instance_generator.max_num_items - 1),
            name="action",
        )

    def _action_mask(
        self,
        obs_ems: EMS,
        obs_ems_mask: chex.Array,
        items: Item,
        items_mask: chex.Array,
        items_placed: chex.Array,
    ) -> chex.Array:
        """Compute the action mask.

        Args:
            obs_ems: tree of ems from the observation.
            obs_ems_mask: mask of ems.
            items: all items.
            items_mask: mask of items.
            items_placed: placing mask of items.

        Returns:
            action_mask: jax array (bool) of shape (obs_num_ems, max_num_items,).
        """
        # TODO
        return jnp.ones((obs_ems_mask.shape[0], items_mask.shape[0]), bool)

    def render(self, state: State) -> Any:
        """Render frames of the environment for a given state.

        Args:
            state: State object containing the current dynamics of the environment.

        Returns:
            Any.

        """
        raise NotImplementedError(
            "Render method is not implemented yet for the BinPack environment. "
            "It will soon be added."
        )

    def close(self) -> None:
        """Perform any necessary cleanup.

        Environments will automatically :meth:`close()` themselves when
        garbage collected or when the program exits.
        """
        pass
