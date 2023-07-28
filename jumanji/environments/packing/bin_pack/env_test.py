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

from typing import Callable, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pytest import FixtureRequest

from jumanji import tree_utils
from jumanji.environments.packing.bin_pack.env import BinPack, ExtendedBinPack
from jumanji.environments.packing.bin_pack.generator import (
    ExtendedRandomGenerator,
    ExtendedToyGenerator,
    RandomGenerator,
    ToyGenerator,
)
from jumanji.environments.packing.bin_pack.space import Space
from jumanji.environments.packing.bin_pack.types import (
    Observation,
    State,
    ValuedItem,
    item_from_space,
    location_from_space,
)
from jumanji.testing.env_not_smoke import (
    SelectActionFn,
    check_env_does_not_smoke,
    check_env_specs_does_not_smoke,
)
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.types import TimeStep


def assert_type_bin_pack_state(state: State) -> None:
    """Assert that all spaces or items are integers while all masks are boolean in the state."""
    jax.tree_util.tree_map(
        lambda leaf: chex.assert_type(leaf, jnp.int32),
        (
            state.container,
            state.ems,
            state.items,
            state.items_location,
            state.sorted_ems_indexes,
        ),
    )
    jax.tree_util.tree_map(
        lambda leaf: chex.assert_type(leaf, bool),
        (state.ems_mask, state.items_mask, state.items_placed, state.action_mask),
    )


def assert_type_extended_bin_pack_state(state: State) -> None:
    """Assert that all spaces or items are integers while all masks are boolean in the state."""
    jax.tree_util.tree_map(
        lambda leaf: chex.assert_type(leaf, jnp.int32),
        (
            state.container,
            state.ems,
            state.items.x_len,
            state.items.y_len,
            state.items.z_len,
            state.items_location,
            state.sorted_ems_indexes,
        ),
    )
    if len(state.items) == 4:
        state.items = cast(ValuedItem, state.items)
        jax.tree_util.tree_map(
            lambda leaf: chex.assert_type(leaf, float),
            (state.items.value,),
        )
    jax.tree_util.tree_map(
        lambda leaf: chex.assert_type(leaf, bool),
        (state.ems_mask, state.items_mask, state.items_placed, state.action_mask),
    )


@pytest.fixture
def bin_pack_random_select_action(bin_pack: BinPack) -> SelectActionFn:
    num_ems, num_items = np.asarray(bin_pack.action_spec.num_values)

    def select_action(key: chex.PRNGKey, observation: Observation) -> chex.Array:
        """Randomly sample valid actions, as determined by `observation.action_mask`."""
        ems_item_id = jax.random.choice(
            key=key,
            a=num_ems * num_items,
            p=observation.action_mask.flatten(),
        )
        ems_id, item_id = jnp.divmod(ems_item_id, num_items)
        action = jnp.array([ems_id, item_id], jnp.int32)
        return action

    return jax.jit(select_action)  # type: ignore


@pytest.fixture(scope="function")
def normalize_dimensions(request: pytest.mark.FixtureRequest) -> bool:
    return request.param  # type: ignore


@pytest.fixture(scope="function")  # noqa: CCR001
def bin_pack_optimal_policy_select_action(  # noqa: CCR001
    request: pytest.mark.FixtureRequest,
) -> Callable[[Observation, State], chex.Array]:
    """Optimal policy for the BinPack environment.
    WARNING: Requires `normalize_dimensions` from the BinPack environment.
    """
    normalize_dimensions = request.node.funcargs["normalize_dimensions"]

    def unnormalize_obs_ems(obs_ems: Space, solution: State) -> Space:
        x_len, y_len, z_len = item_from_space(solution.container)
        norm_space = Space(x1=x_len, x2=x_len, y1=y_len, y2=y_len, z1=z_len, z2=z_len)
        obs_ems: Space = jax.tree_util.tree_map(
            lambda x, c: jnp.round(x * c).astype(jnp.int32),
            obs_ems,
            norm_space,
        )
        return obs_ems

    def select_action(  # noqa: CCR001
        observation: Observation, solution: State
    ) -> chex.Array:
        """Outputs the best action to fully pack the container."""
        for obs_ems_id, obs_ems_action_mask in enumerate(observation.action_mask):
            if not obs_ems_action_mask.any():
                continue
            obs_ems = tree_utils.tree_slice(observation.ems, obs_ems_id)
            if normalize_dimensions:
                obs_ems = unnormalize_obs_ems(obs_ems, solution)
            obs_ems_location = location_from_space(obs_ems)
            for item_id, action_feasible in enumerate(obs_ems_action_mask):
                if not action_feasible:
                    continue
                item_location = tree_utils.tree_slice(solution.items_location, item_id)
                if item_location == obs_ems_location:
                    return jnp.array([obs_ems_id, item_id], jnp.int32)
        raise LookupError("Could not find the optimal action.")

    return select_action


def test_bin_pack__reset(bin_pack: BinPack) -> None:
    """Validates the jitted reset of the environment."""
    chex.clear_trace_counter()
    reset_fn = jax.jit(chex.assert_max_traces(bin_pack.reset, n=1))

    key = jax.random.PRNGKey(0)
    _ = reset_fn(key)
    # Call again to check it does not compile twice.
    state, timestep = reset_fn(key)
    assert isinstance(timestep, TimeStep)
    assert isinstance(state, State)
    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # reset function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(state)
    assert_type_bin_pack_state(state)
    assert state.ems_mask.any()
    assert state.items_mask.any()
    assert jnp.any(state.action_mask)


def test_bin_pack_step__jit(bin_pack: BinPack) -> None:
    """Validates jitting the environment step function."""
    chex.clear_trace_counter()
    step_fn = jax.jit(chex.assert_max_traces(bin_pack.step, n=1))

    key = jax.random.PRNGKey(0)
    state, timestep = bin_pack.reset(key)

    action = bin_pack.action_spec.generate_value()
    _ = step_fn(state, action)
    # Call again to check it does not compile twice.
    state, timestep = step_fn(state, action)
    assert_type_bin_pack_state(state)


def test_bin_pack__render_does_not_smoke(bin_pack: BinPack, dummy_state: State) -> None:
    bin_pack.render(dummy_state)
    bin_pack.close()


def test_bin_pack__does_not_smoke(
    bin_pack: BinPack,
    bin_pack_random_select_action: SelectActionFn,
) -> None:
    """Test that we can run an episode without any errors."""
    check_env_does_not_smoke(bin_pack, bin_pack_random_select_action)


def test_bin_pack__specs_does_not_smoke(bin_pack: BinPack) -> None:
    """Test that we can access specs without any errors."""
    check_env_specs_does_not_smoke(bin_pack)


def test_bin_pack__pack_all_items_dummy_instance(
    bin_pack: BinPack, bin_pack_random_select_action: SelectActionFn
) -> None:
    """Functional test to check that the dummy instance can be completed with a random agent."""
    step_fn = jax.jit(bin_pack.step)
    key = jax.random.PRNGKey(0)
    state, timestep = bin_pack.reset(key)

    while not timestep.last():
        action_key, key = jax.random.split(key)
        action = bin_pack_random_select_action(action_key, timestep.observation)
        state, timestep = step_fn(state, action)

    assert jnp.array_equal(state.items_placed, state.items_mask)


@pytest.mark.parametrize("normalize_dimensions", [False, True], indirect=True)
def test_bin_pack__optimal_policy_toy_instance(
    normalize_dimensions: bool,
    bin_pack_optimal_policy_select_action: Callable[[Observation, State], chex.Array],
    toy_generator: ToyGenerator,
) -> None:
    """Functional test to check that the toy instance can be optimally packed with an optimal
    policy. Checks for both options: normalizing dimensions and not normalizing.
    """
    toy_bin_pack = BinPack(
        generator=toy_generator,
        obs_num_ems=40,
        normalize_dimensions=normalize_dimensions,
        debug=True,
    )
    key = jax.random.PRNGKey(0)
    step_fn = jax.jit(toy_bin_pack.step)
    state, timestep = toy_bin_pack.reset(key)
    solution = toy_bin_pack.generator.generate_solution(key)

    while not timestep.last():
        action = bin_pack_optimal_policy_select_action(timestep.observation, solution)
        state, timestep = step_fn(state, action)
        assert isinstance(timestep.extras, dict)
        assert not timestep.extras["invalid_action"]
        assert not timestep.extras["invalid_ems_from_env"]
    assert jnp.array_equal(state.items_placed, solution.items_placed)


@pytest.mark.parametrize(
    "normalize_dimensions, max_num_items, max_num_ems, obs_num_ems",
    [
        (False, 5, 20, 10),
        (True, 5, 20, 10),
        (False, 20, 80, 50),
        (True, 20, 80, 50),
    ],
)
def test_bin_pack__optimal_policy_random_instance(
    normalize_dimensions: bool,
    bin_pack_optimal_policy_select_action: Callable[[Observation, State], chex.Array],
    max_num_items: int,
    max_num_ems: int,
    obs_num_ems: int,
) -> None:
    """Functional test to check that random instances can be optimally packed with an optimal
    policy. Checks for both options: normalizing dimensions and not normalizing, and checks for
    two different sizes: 5 items and 20 items, with respectively 20 and 80 max number of EMSs.
    """
    num_trial_episodes = 3
    random_bin_pack = BinPack(
        generator=RandomGenerator(max_num_items, max_num_ems),
        obs_num_ems=obs_num_ems,
        normalize_dimensions=normalize_dimensions,
        debug=True,
    )
    reset_fn = jax.jit(random_bin_pack.reset)
    generate_solution_fn = jax.jit(random_bin_pack.generator.generate_solution)
    step_fn = jax.jit(random_bin_pack.step)
    for key in jax.random.split(jax.random.PRNGKey(0), num_trial_episodes):
        state, timestep = reset_fn(key)
        solution = generate_solution_fn(key)

        while not timestep.last():
            action = bin_pack_optimal_policy_select_action(
                timestep.observation, solution
            )
            assert timestep.observation.action_mask[tuple(action)]
            state, timestep = step_fn(state, action)
            assert not timestep.extras["invalid_action"]
            assert not timestep.extras["invalid_ems_from_env"]
        assert jnp.array_equal(state.items_placed, solution.items_placed)


def test_full_support_bin_pack(full_support_bin_pack: BinPack) -> None:
    """
    This test checks that no unsupported item can be placed by the agent
    and that the merging of EMS with the same height works correctly.
    """
    step_fn = jax.jit(full_support_bin_pack.step)
    state, timestep = jax.jit(full_support_bin_pack.reset)(0)
    # Start by forcing the agent to place one of the small items.
    state, timestep = step_fn(state, jnp.array([0, 1]))
    nb_remaning_items = full_support_bin_pack.generator.max_num_items - 1
    while not timestep.last() and nb_remaning_items > 1:
        # Trick to select the EMS whose bottom is the floor of the container (when the number of
        # items is less than 6 the biggest ems becomes the EMS whose bottom is the top face of
        # the set of placed items).
        # At first int(nb_remaning_items < 6) = 0 since the number of reamining items is bigger than
        # 6 and so the EMS selected would be the biggest EMS which is the one whose bottom side is
        # the floor. When the number of remaining items is less than 6 int(nb_remaning_items < 6)
        # would be equal to 1 and so the selected EMS would be the second largest EMS which is
        # the one with the bottom side on the floor.
        action = jnp.array([int(nb_remaning_items < 6), nb_remaning_items])
        assert timestep.observation.action_mask[tuple(action)]
        # Make sure that the big item can't be placed because it won't be supported
        assert jnp.all(~timestep.observation.action_mask[:, 0])
        state, timestep = step_fn(state, action)
        # Make sure that big piece isn't placeable because it can't be fully supported.
        assert not timestep.extras["invalid_action"]
        assert not timestep.extras["invalid_ems_from_env"]
        nb_remaning_items -= 1
    action = jnp.array([0, 0])
    assert timestep.observation.action_mask[tuple(action)]
    state, timestep = step_fn(state, action)

    # Make sure that all the items were placed and that the container was filled to 100%
    assert jnp.array_equal(state.items_placed, jnp.array(11 * [True]))
    assert jnp.isclose(timestep.extras["volume_utilization"], 1)


@pytest.mark.parametrize(
    "normalize_dimensions, max_num_items, max_num_ems, obs_num_ems",
    [
        (False, 5, 20, 10),
        (True, 5, 20, 10),
        (False, 20, 80, 50),
        (True, 20, 80, 50),
    ],
)
def test_full_support_bin_pack__optimal_policy_random_instance(
    normalize_dimensions: bool,
    bin_pack_optimal_policy_select_action: Callable[[Observation, State], chex.Array],
    max_num_items: int,
    max_num_ems: int,
    obs_num_ems: int,
) -> None:
    """Functional test to check that random instances can be optimally packed with an optimal
    policy. Checks for both options: normalizing dimensions and not normalizing, and checks for
    two different sizes: 5 items and 20 items, with respectively 20 and 80 max number of EMSs.
    """
    num_trial_episodes = 3
    random_bin_pack = BinPack(
        generator=RandomGenerator(max_num_items, max_num_ems),
        obs_num_ems=obs_num_ems,
        normalize_dimensions=normalize_dimensions,
        debug=True,
        full_support=True,
    )
    reset_fn = jax.jit(random_bin_pack.reset)
    generate_solution_fn = jax.jit(random_bin_pack.generator.generate_solution)
    step_fn = jax.jit(random_bin_pack.step)
    for key in jax.random.split(jax.random.PRNGKey(0), num_trial_episodes):
        state, timestep = reset_fn(key)
        solution = generate_solution_fn(key)

        while not timestep.last():
            action = bin_pack_optimal_policy_select_action(
                timestep.observation, solution
            )
            assert timestep.observation.action_mask[tuple(action)]
            state, timestep = step_fn(state, action)
            assert not timestep.extras["invalid_action"]
            assert not timestep.extras["invalid_ems_from_env"]
        assert jnp.array_equal(state.items_placed, solution.items_placed)


class TestExtendedBinPackRotationNoValue:

    """
    Class Used to test the Extended Bin pack environment when Items are allowed to take all
    possible orientations but have no value associated to them.
    """

    @pytest.fixture
    def rotation_bin_pack_random_select_action(
        self, rotation_bin_pack: ExtendedBinPack
    ) -> SelectActionFn:
        num_orientations, num_ems, num_items = np.asarray(
            rotation_bin_pack.action_spec().num_values
        )

        def select_action(key: chex.PRNGKey, observation: Observation) -> chex.Array:
            """Randomly sample valid actions, as determined by `observation.action_mask`."""
            orientation_ems_item_id = jax.random.choice(
                key=key,
                a=num_orientations * num_ems * num_items,
                p=observation.action_mask.flatten(),
            )
            orientation_ems_id, item_id = jnp.divmod(orientation_ems_item_id, num_items)
            orientation, ems_id = jnp.divmod(orientation_ems_id, num_ems)
            action = jnp.array([orientation, ems_id, item_id], jnp.int32)
            return action

        return jax.jit(select_action)  # type: ignore

    @pytest.fixture  # noqa: CCR001
    def rotation_bin_pack_optimal_policy_select_action(  # noqa: CCR001
        self,
        request: FixtureRequest,
    ) -> Callable[[Observation, State], chex.Array]:
        """Optimal policy for the BinPack environment.
        WARNING: Requires `normalize_dimensions` from the BinPack environment.
        """
        normalize_dimensions = request.param

        def unnormalize_obs_ems(obs_ems: Space, solution: State) -> Space:
            x_len, y_len, z_len = item_from_space(solution.container)
            norm_space = Space(
                x1=x_len, x2=x_len, y1=y_len, y2=y_len, z1=z_len, z2=z_len
            )
            obs_ems: Space = jax.tree_util.tree_map(
                lambda x, c: jnp.round(x * c).astype(jnp.int32),
                obs_ems,
                norm_space,
            )
            return obs_ems

        def select_action(  # noqa: CCR001
            observation: Observation, solution: State
        ) -> chex.Array:
            """Outputs the best action to fully pack the container."""
            reshaped_action_mask = observation.action_mask.reshape(
                6, observation.action_mask.shape[0], -1
            )
            for obs_ems_id, obs_ems_action_mask in enumerate(reshaped_action_mask[0]):
                if not obs_ems_action_mask.any():
                    continue
                obs_ems = tree_utils.tree_slice(observation.ems, obs_ems_id)
                if normalize_dimensions:
                    obs_ems = unnormalize_obs_ems(obs_ems, solution)
                obs_ems_location = location_from_space(obs_ems)
                for item_id, action_feasible in enumerate(obs_ems_action_mask):
                    if not action_feasible:
                        continue
                    item_location = tree_utils.tree_slice(
                        solution.items_location, item_id
                    )
                    if item_location == obs_ems_location:
                        return jnp.array([0, obs_ems_id, item_id], jnp.int32)
            raise LookupError("Could not find the optimal action.")

        return select_action

    def test__rotation_bin_pack__reset(
        self, rotation_bin_pack: ExtendedBinPack
    ) -> None:
        """Validates the jitted reset of the environment."""
        chex.clear_trace_counter()
        reset_fn = jax.jit(chex.assert_max_traces(rotation_bin_pack.reset, n=1))

        key = jax.random.PRNGKey(0)
        _ = reset_fn(key)
        # Call again to check it does not compile twice.
        state, timestep = reset_fn(key)
        assert isinstance(timestep, TimeStep)
        assert isinstance(state, State)
        # Check that the state is made of DeviceArrays, this is false for the non-jitted
        # reset function since unpacking random.split returns numpy arrays and not device arrays.
        assert_is_jax_array_tree(state)
        assert_type_bin_pack_state(state)
        assert state.ems_mask.any()
        assert jnp.any(state.action_mask)
        assert state.items_mask.any()

    def test_rotation_bin_pack_step__jit(self, rotation_bin_pack: BinPack) -> None:
        """Validates jitting the environment step function."""
        chex.clear_trace_counter()
        step_fn = jax.jit(chex.assert_max_traces(rotation_bin_pack.step, n=1))

        key = jax.random.PRNGKey(0)
        state, timestep = rotation_bin_pack.reset(key)

        action = rotation_bin_pack.action_spec().generate_value()
        _ = step_fn(state, action)
        # Call again to check it does not compile twice.
        state, timestep = step_fn(state, action)
        assert_type_bin_pack_state(state)

    def test_bin_pack__render_does_not_smoke(
        self, rotation_bin_pack: ExtendedBinPack, dummy_rotation_state: State
    ) -> None:
        rotation_bin_pack.render(dummy_rotation_state)
        rotation_bin_pack.close()

    def test_bin_pack__does_not_smoke(
        self,
        rotation_bin_pack: ExtendedBinPack,
        rotation_bin_pack_random_select_action: SelectActionFn,
    ) -> None:
        """Test that we can run an episode without any errors."""
        check_env_does_not_smoke(
            rotation_bin_pack, rotation_bin_pack_random_select_action
        )

    def test_bin_pack__pack_all_items_dummy_instance(
        self,
        rotation_bin_pack: ExtendedBinPack,
        rotation_bin_pack_random_select_action: SelectActionFn,
    ) -> None:
        """Functional test to check that the dummy instance can be completed with a random agent."""
        step_fn = jax.jit(rotation_bin_pack.step)
        key = jax.random.PRNGKey(0)
        state, timestep = rotation_bin_pack.reset(key)

        while not timestep.last():
            action_key, key = jax.random.split(key)
            action = rotation_bin_pack_random_select_action(
                action_key, timestep.observation
            )
            state, timestep = step_fn(state, action)

        assert jnp.array_equal(jnp.sum(state.items_placed), state.nb_items)

    @pytest.mark.parametrize(
        "rotation_bin_pack_optimal_policy_select_action, normalize_dimensions",
        [(False, False), (True, True)],
        indirect=["rotation_bin_pack_optimal_policy_select_action"],
    )
    def test_bin_pack__optimal_policy_toy_instance(
        self,
        rotation_bin_pack_optimal_policy_select_action: Callable[
            [Observation, State], chex.Array
        ],
        rotation_toy_generator: ExtendedToyGenerator,
        normalize_dimensions: bool,
    ) -> None:
        """Functional test to check that the toy instance can be optimally packed with an optimal
        policy. Checks for both options: normalizing dimensions and not normalizing.
        """
        toy_bin_pack = ExtendedBinPack(
            generator=rotation_toy_generator,
            obs_num_ems=40,
            normalize_dimensions=normalize_dimensions,
            debug=True,
            is_rotation_allowed=True,
            is_value_based=False,
        )
        key = jax.random.PRNGKey(0)
        step_fn = jax.jit(toy_bin_pack.step)
        state, timestep = toy_bin_pack.reset(key)
        solution = toy_bin_pack.generator.generate_solution(key)

        while not timestep.last():
            action = rotation_bin_pack_optimal_policy_select_action(
                timestep.observation, solution
            )
            state, timestep = step_fn(state, action)
            assert isinstance(timestep.extras, dict)
            # This is not true anymore since there are items that can't
            # fit in all the possible orientations
            # assert not timestep.extras["invalid_action"]
            assert not jnp.any(timestep.extras["invalid_ems_from_env"])
        if timestep.extras is not None:
            assert timestep.extras["volume_utilization"] == 1
            assert timestep.extras["ratio_packed_items"] == 1

    @pytest.mark.parametrize(
        "rotation_bin_pack_optimal_policy_select_action, \
            normalize_dimensions, max_num_items, max_num_ems, obs_num_ems",
        [
            (False, False, 5, 20, 10),
            (True, True, 5, 20, 10),
            (False, False, 20, 80, 50),
            (True, True, 20, 80, 50),
        ],
        indirect=["rotation_bin_pack_optimal_policy_select_action"],
    )
    def test_bin_pack__optimal_policy_random_instance(
        self,
        rotation_bin_pack_optimal_policy_select_action: Callable[
            [Observation, State], chex.Array
        ],
        normalize_dimensions: bool,
        max_num_items: int,
        max_num_ems: int,
        obs_num_ems: int,
    ) -> None:
        """Functional test to check that random instances can be optimally packed with an optimal
        policy. Checks for both options: normalizing dimensions and not normalizing, and checks for
        two different sizes: 5 items and 20 items, with respectively 20 and 80 max number of EMSs.
        """
        num_trial_episodes = 3
        random_bin_pack = ExtendedBinPack(
            generator=ExtendedRandomGenerator(
                max_num_items,
                max_num_ems,
                is_rotation_allowed=True,
                is_value_based=False,
            ),
            obs_num_ems=obs_num_ems,
            normalize_dimensions=normalize_dimensions,
            debug=True,
            is_rotation_allowed=True,
            is_value_based=False,
        )
        reset_fn = jax.jit(random_bin_pack.reset)
        generate_solution_fn = jax.jit(random_bin_pack.generator.generate_solution)
        step_fn = jax.jit(random_bin_pack.step)
        for key in jax.random.split(jax.random.PRNGKey(0), num_trial_episodes):
            state, timestep = reset_fn(key)
            solution = generate_solution_fn(key)

            while not timestep.last():
                action = rotation_bin_pack_optimal_policy_select_action(
                    timestep.observation, solution
                )
                reshaped_action_mask = timestep.observation.action_mask.reshape(
                    6, timestep.observation.action_mask.shape[0], -1
                )
                assert reshaped_action_mask[tuple(action)]
                state, timestep = step_fn(state, action)
                # assert not timestep.extras["invalid_action"]
                assert not jnp.any(timestep.extras["invalid_ems_from_env"])
            assert jnp.array_equal(state.items_placed, solution.items_placed)
            assert round(timestep.extras["volume_utilization"]) == 1
            assert timestep.extras["ratio_packed_items"] == 1


class TestExtendedBinPackRotationValue:
    """
    Class Used to test the Extended Bin pack environment when Items are allowed to take all
    possible orientations and have a value associated with them.
    """

    @pytest.fixture
    def extended_bin_pack_random_select_action(
        self, extended_bin_pack: ExtendedBinPack
    ) -> SelectActionFn:
        num_orientations, num_ems, num_items = np.asarray(
            extended_bin_pack.action_spec().num_values
        )

        def select_action(key: chex.PRNGKey, observation: Observation) -> chex.Array:
            """Randomly sample valid actions, as determined by `observation.action_mask`."""
            orientation_ems_item_id = jax.random.choice(
                key=key,
                a=num_orientations * num_ems * num_items,
                p=observation.action_mask.flatten(),
            )
            orientation_ems_id, item_id = jnp.divmod(orientation_ems_item_id, num_items)
            orientation, ems_id = jnp.divmod(orientation_ems_id, num_ems)
            action = jnp.array([orientation, ems_id, item_id], jnp.int32)
            return action

        return jax.jit(select_action)  # type: ignore

    @pytest.fixture  # noqa: CCR001
    def extended_bin_pack_optimal_policy_select_action(  # noqa: CCR001
        self,
        request: FixtureRequest,
    ) -> Callable[[Observation, State], chex.Array]:
        """Optimal policy for the BinPack environment.
        WARNING: Requires `normalize_dimensions` from the BinPack environment.
        """
        normalize_dimensions = request.param

        def unnormalize_obs_ems(obs_ems: Space, solution: State) -> Space:
            x_len, y_len, z_len = item_from_space(solution.container)
            norm_space = Space(
                x1=x_len, x2=x_len, y1=y_len, y2=y_len, z1=z_len, z2=z_len
            )
            obs_ems: Space = jax.tree_util.tree_map(
                lambda x, c: jnp.round(x * c).astype(jnp.int32),
                obs_ems,
                norm_space,
            )
            return obs_ems

        def select_action(  # noqa: CCR001
            observation: Observation, solution: State
        ) -> chex.Array:
            """Outputs the best action to fully pack the container."""
            reshaped_action_mask = observation.action_mask.reshape(
                6, observation.action_mask.shape[0], -1
            )
            for obs_ems_id, obs_ems_action_mask in enumerate(reshaped_action_mask[0]):
                if not obs_ems_action_mask.any():
                    continue
                obs_ems = tree_utils.tree_slice(observation.ems, obs_ems_id)
                if normalize_dimensions:
                    obs_ems = unnormalize_obs_ems(obs_ems, solution)
                obs_ems_location = location_from_space(obs_ems)
                for item_id, action_feasible in enumerate(obs_ems_action_mask):
                    if not action_feasible:
                        continue
                    item_location = tree_utils.tree_slice(
                        solution.items_location, item_id
                    )
                    if item_location == obs_ems_location:
                        return jnp.array([0, obs_ems_id, item_id], jnp.int32)
            raise LookupError("Could not find the optimal action.")

        return select_action

    def test__extended_bin_pack__reset(
        self, extended_bin_pack: ExtendedBinPack
    ) -> None:
        """Validates the jitted reset of the environment."""
        chex.clear_trace_counter()
        reset_fn = jax.jit(chex.assert_max_traces(extended_bin_pack.reset, n=1))

        key = jax.random.PRNGKey(0)
        _ = reset_fn(key)
        # Call again to check it does not compile twice.
        state, timestep = reset_fn(key)
        assert isinstance(timestep, TimeStep)
        assert isinstance(state, State)
        # Check that the state is made of DeviceArrays, this is false for the non-jitted
        # reset function since unpacking random.split returns numpy arrays and not device arrays.
        assert_is_jax_array_tree(state)
        assert_type_extended_bin_pack_state(state)
        assert state.ems_mask.any()
        assert jnp.any(state.action_mask)
        assert state.items_mask.any()

    def test_extended_bin_pack_step__jit(self, extended_bin_pack: BinPack) -> None:
        """Validates jitting the environment step function."""
        chex.clear_trace_counter()
        step_fn = jax.jit(chex.assert_max_traces(extended_bin_pack.step, n=1))

        key = jax.random.PRNGKey(0)
        state, timestep = extended_bin_pack.reset(key)

        action = extended_bin_pack.action_spec().generate_value()
        _ = step_fn(state, action)
        # Call again to check it does not compile twice.
        state, timestep = step_fn(state, action)
        assert_type_extended_bin_pack_state(state)

    def test_bin_pack__render_does_not_smoke(
        self, extended_bin_pack: ExtendedBinPack, dummy_rotation_state: State
    ) -> None:
        extended_bin_pack.render(dummy_rotation_state)
        extended_bin_pack.close()

    def test_bin_pack__does_not_smoke(
        self,
        extended_bin_pack: ExtendedBinPack,
        extended_bin_pack_random_select_action: SelectActionFn,
    ) -> None:
        """Test that we can run an episode without any errors."""
        check_env_does_not_smoke(
            extended_bin_pack, extended_bin_pack_random_select_action
        )

    def test_bin_pack__pack_all_items_dummy_instance(
        self,
        extended_bin_pack: ExtendedBinPack,
        extended_bin_pack_random_select_action: SelectActionFn,
    ) -> None:
        """Functional test to check that the dummy instance can be completed with a random agent."""
        step_fn = jax.jit(extended_bin_pack.step)
        key = jax.random.PRNGKey(0)
        state, timestep = extended_bin_pack.reset(key)

        while not timestep.last():
            action_key, key = jax.random.split(key)
            action = extended_bin_pack_random_select_action(
                action_key, timestep.observation
            )
            state, timestep = step_fn(state, action)

        assert jnp.array_equal(jnp.sum(state.items_placed), state.nb_items)

    @pytest.mark.parametrize(
        "extended_bin_pack_optimal_policy_select_action, normalize_dimensions",
        [(False, False), (True, True)],
        indirect=["extended_bin_pack_optimal_policy_select_action"],
    )
    def test_bin_pack__optimal_policy_toy_instance(
        self,
        extended_bin_pack_optimal_policy_select_action: Callable[
            [Observation, State], chex.Array
        ],
        rotation_toy_generator: ExtendedToyGenerator,
        normalize_dimensions: bool,
    ) -> None:
        """Functional test to check that the toy instance can be optimally packed with an optimal
        policy. Checks for both options: normalizing dimensions and not normalizing.
        """
        toy_bin_pack = ExtendedBinPack(
            generator=rotation_toy_generator,
            obs_num_ems=40,
            normalize_dimensions=normalize_dimensions,
            debug=True,
            is_rotation_allowed=True,
            is_value_based=False,
        )
        key = jax.random.PRNGKey(0)
        step_fn = jax.jit(toy_bin_pack.step)
        state, timestep = toy_bin_pack.reset(key)
        solution = toy_bin_pack.generator.generate_solution(key)

        while not timestep.last():
            action = extended_bin_pack_optimal_policy_select_action(
                timestep.observation, solution
            )
            state, timestep = step_fn(state, action)
            assert isinstance(timestep.extras, dict)
            # This is not true anymore since there are items that can't
            # fit in all the possible orientations
            # assert not timestep.extras["invalid_action"]
            assert not jnp.any(timestep.extras["invalid_ems_from_env"])
        if timestep.extras is not None:
            assert timestep.extras["volume_utilization"] == 1
            assert timestep.extras["ratio_packed_items"] == 1

    @pytest.mark.parametrize(
        "extended_bin_pack_optimal_policy_select_action, \
            normalize_dimensions, max_num_items, max_num_ems, obs_num_ems",
        [
            (False, False, 5, 20, 10),
            (True, True, 5, 20, 10),
            (False, False, 20, 80, 50),
            (True, True, 20, 80, 50),
        ],
        indirect=["extended_bin_pack_optimal_policy_select_action"],
    )
    def test_bin_pack__optimal_policy_random_instance(
        self,
        extended_bin_pack_optimal_policy_select_action: Callable[
            [Observation, State], chex.Array
        ],
        normalize_dimensions: bool,
        max_num_items: int,
        max_num_ems: int,
        obs_num_ems: int,
    ) -> None:
        """Functional test to check that random instances can be optimally packed with an optimal
        policy. Checks for both options: normalizing dimensions and not normalizing, and checks for
        two different sizes: 5 items and 20 items, with respectively 20 and 80 max number of EMSs.
        """
        num_trial_episodes = 3
        random_bin_pack = ExtendedBinPack(
            generator=ExtendedRandomGenerator(
                max_num_items,
                max_num_ems,
                is_rotation_allowed=True,
                is_value_based=True,
                mean_item_value=1,
                std_item_value=0.5,
            ),
            obs_num_ems=obs_num_ems,
            normalize_dimensions=normalize_dimensions,
            debug=True,
            is_rotation_allowed=True,
            is_value_based=False,
        )
        reset_fn = jax.jit(random_bin_pack.reset)
        generate_solution_fn = jax.jit(random_bin_pack.generator.generate_solution)
        step_fn = jax.jit(random_bin_pack.step)
        for key in jax.random.split(jax.random.PRNGKey(0), num_trial_episodes):
            state, timestep = reset_fn(key)
            solution = generate_solution_fn(key)

            while not timestep.last():
                action = extended_bin_pack_optimal_policy_select_action(
                    timestep.observation, solution
                )
                reshaped_action_mask = timestep.observation.action_mask.reshape(
                    6, timestep.observation.action_mask.shape[0], -1
                )
                assert reshaped_action_mask[tuple(action)]
                state, timestep = step_fn(state, action)
                # assert not timestep.extras["invalid_action"]
                assert not jnp.any(timestep.extras["invalid_ems_from_env"])
            assert jnp.array_equal(state.items_placed, solution.items_placed)
            assert round(timestep.extras["volume_utilization"]) == 1
