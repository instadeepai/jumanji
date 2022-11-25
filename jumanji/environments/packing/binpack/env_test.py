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

from typing import Callable, Type

import chex
import jax
import jax.numpy as jnp
import pytest

from jumanji import specs, tree_utils
from jumanji.environments.packing.binpack.env import BinPack
from jumanji.environments.packing.binpack.instance_generator import (
    InstanceGenerator,
    RandomInstanceGenerator,
    ToyInstanceGenerator,
)
from jumanji.environments.packing.binpack.space import Space
from jumanji.environments.packing.binpack.specs import (
    EMSSpec,
    ItemSpec,
    ObservationSpec,
)
from jumanji.environments.packing.binpack.types import (
    Observation,
    State,
    location_from_space,
)
from jumanji.testing.env_not_smoke import SelectActionFn, check_env_does_not_smoke
from jumanji.testing.pytrees import assert_is_jax_array_tree, assert_trees_are_equal
from jumanji.types import TimeStep


@pytest.fixture
def binpack_random_select_action(binpack_env: BinPack) -> SelectActionFn:
    num_ems, num_items = binpack_env.action_spec().num_values

    def select_action(rng_key: chex.PRNGKey, obs: Observation) -> chex.Array:
        """Randomly sample valid actions, as determined by `observation.action_mask`."""
        ems_item_id = jax.random.choice(
            key=rng_key,
            a=num_ems * num_items,
            p=obs.action_mask.flatten(),
        )
        ems_id, item_id = jnp.divmod(ems_item_id, num_items)
        action = jnp.array([ems_id, item_id], int)
        return action

    return select_action


@pytest.fixture(scope="function")
def normalize_dimensions(request: pytest.mark.FixtureRequest) -> bool:
    return request.param  # type: ignore


@pytest.fixture(scope="function")
def binpack_optimal_policy_select_action(  # noqa: CCR001
    request: pytest.mark.FixtureRequest,
) -> Callable[[Observation, State], chex.Array]:
    """Optimal policy for the BinPack environment.
    WARNING: Requires `normalize_dimensions` from the BinPack environment.
    """
    normalize_dimensions = request.node.funcargs["normalize_dimensions"]

    def unnormalize_obs_ems(obs_ems: Space, solution: State) -> Space:
        container_space_to_normalize = Space(
            x1=solution.container.x2,
            x2=solution.container.x2,
            y1=solution.container.y2,
            y2=solution.container.y2,
            z1=solution.container.z2,
            z2=solution.container.z2,
        )
        obs_ems: Space = jax.tree_map(
            lambda x, c: jnp.round(x * c).astype(jnp.int32),
            obs_ems,
            container_space_to_normalize,
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
                    return jnp.array([obs_ems_id, item_id], int)
        raise LookupError("Could not find the optimal action.")

    return select_action


def assert_type_binpack_state(state: State) -> None:
    """Assert that all spaces or items are integers while all masks are boolean in the state."""
    jax.tree_map(
        lambda leaf: chex.assert_type(leaf, int),
        (
            state.container,
            state.ems,
            state.items,
            state.items_location,
            state.sorted_ems_indexes,
        ),
    )
    jax.tree_map(
        lambda leaf: chex.assert_type(leaf, bool),
        (state.ems_mask, state.items_mask, state.items_placed, state.action_mask),
    )


def test_binpack__reset(binpack_env: BinPack) -> None:
    """Validates the jitted reset of the environment."""
    chex.clear_trace_counter()
    reset_fn = jax.jit(chex.assert_max_traces(binpack_env.reset, n=1))

    key = jax.random.PRNGKey(0)
    state, timestep = reset_fn(key)
    # Call again to check it does not compile twice.
    state, timestep = reset_fn(key)
    assert isinstance(timestep, TimeStep)
    assert isinstance(state, State)
    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # reset function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(state)
    assert_type_binpack_state(state)
    assert state.ems_mask.any()
    assert state.items_mask.any()
    assert jnp.any(state.action_mask)


@pytest.mark.parametrize("normalize_dimensions", [True, False])
def test_binpack__spec(
    normalize_dimensions: bool, dummy_instance_generator: InstanceGenerator
) -> None:
    """Validates the observation and action spec of the BinPack environment. Checks that
    different specs are generated depending on the `normalize_dimensions` argument.
    """
    binpack_env = BinPack(
        instance_generator=dummy_instance_generator,
        obs_num_ems=1,
        normalize_dimensions=normalize_dimensions,
    )
    observation_spec = binpack_env.observation_spec()
    assert isinstance(observation_spec, ObservationSpec)
    assert isinstance(observation_spec.ems_spec, EMSSpec)
    assert isinstance(observation_spec.ems_mask_spec, specs.BoundedArray)
    assert isinstance(observation_spec.items_spec, ItemSpec)
    assert isinstance(observation_spec.items_mask_spec, specs.BoundedArray)
    assert isinstance(observation_spec.items_placed_spec, specs.BoundedArray)
    assert isinstance(observation_spec.action_mask_spec, specs.BoundedArray)
    assert isinstance(binpack_env.action_spec(), specs.MultiDiscreteArray)
    observation = observation_spec.generate_value()

    def assert_type_binpack_observation(obs: Observation, type_: Type) -> None:
        """Assert that the EMS and items are of given type."""
        jax.tree_map(lambda leaf: chex.assert_type(leaf, type_), (obs.ems, obs.items))

    if normalize_dimensions:
        assert_type_binpack_observation(observation, float)
    else:
        assert_type_binpack_observation(observation, int)


def test_binpack_step__jit(binpack_env: BinPack) -> None:
    """Validates jitting the environment step function."""
    chex.clear_trace_counter()
    step_fn = jax.jit(chex.assert_max_traces(binpack_env.step, n=1))

    key = jax.random.PRNGKey(0)
    state, timestep = binpack_env.reset(key)

    action = binpack_env.action_spec().generate_value()
    state, timestep = step_fn(state, action)
    # Call again to check it does not compile twice.
    state, timestep = step_fn(state, action)
    assert_type_binpack_state(state)


def test_binpack__render_does_not_smoke(
    binpack_env: BinPack, dummy_instance: State
) -> None:
    binpack_env.render(dummy_instance)
    binpack_env.close()


def test_binpack__does_not_smoke(
    binpack_env: BinPack,
    binpack_random_select_action: SelectActionFn,
) -> None:
    """Test that we can run an episode without any errors."""
    check_env_does_not_smoke(binpack_env, binpack_random_select_action)


def test_binpack__pack_all_items_dummy_instance(
    binpack_env: BinPack, binpack_random_select_action: SelectActionFn
) -> None:
    """Functional test to check that the dummy instance can be completed with a random agent."""
    key = jax.random.PRNGKey(0)
    state, timestep = binpack_env.reset(key)

    while not timestep.last():
        action_key, key = jax.random.split(key)
        action = binpack_random_select_action(action_key, timestep.observation)
        state, timestep = jax.jit(binpack_env.step)(state, action)

    assert_trees_are_equal(state.items_placed, state.items_mask)


@pytest.mark.parametrize("normalize_dimensions", [False, True], indirect=True)
def test_binpack__optimal_policy_toy_instance(
    normalize_dimensions: bool,
    binpack_optimal_policy_select_action: Callable[[Observation, State], chex.Array],
) -> None:
    """Functional test to check that the toy instance can be optimally packed with an optimal
    policy. Checks for both options: normalizing dimensions and not normalizing.
    """
    toy_binpack = BinPack(
        ToyInstanceGenerator(),
        obs_num_ems=40,
        normalize_dimensions=normalize_dimensions,
        debug=True,
    )
    key = jax.random.PRNGKey(0)
    step_fn = jax.jit(toy_binpack.step)
    state, timestep = toy_binpack.reset(key)
    solution = toy_binpack.instance_generator.generate_solution(key)

    while not timestep.last():
        action_key, key = jax.random.split(key)
        action = binpack_optimal_policy_select_action(timestep.observation, solution)
        state, timestep = step_fn(state, action)
        assert not timestep.extras["invalid_action_taken"]  # type: ignore
        assert not timestep.extras["invalid_ems_from_env"]  # type: ignore
    assert_trees_are_equal(state.items_placed, solution.items_placed)


@pytest.mark.parametrize(
    "normalize_dimensions, max_num_items, max_num_ems, obs_num_ems",
    [
        (False, 20, 80, 40),
        (True, 20, 80, 40),
        (False, 40, 200, 60),
        (True, 40, 200, 60),
    ],
    indirect=["normalize_dimensions"],
)
def test_binpack__optimal_policy_random_instance(
    normalize_dimensions: bool,
    binpack_optimal_policy_select_action: Callable[[Observation, State], chex.Array],
    max_num_items: int,
    max_num_ems: int,
    obs_num_ems: int,
) -> None:
    """Functional test to check that random instances can be optimally packed with an optimal
    policy. Checks for both options: normalizing dimensions and not normalizing, and checks for
    two different sizes: 20 items and 40 items.
    """
    random_binpack = BinPack(
        RandomInstanceGenerator(max_num_items=max_num_items, max_num_ems=max_num_ems),
        obs_num_ems=obs_num_ems,
        normalize_dimensions=normalize_dimensions,
        debug=True,
    )
    reset_fn = jax.jit(random_binpack.reset)
    generate_solution_fn = jax.jit(random_binpack.instance_generator.generate_solution)
    step_fn = jax.jit(random_binpack.step)
    for seed in range(10):
        key = jax.random.PRNGKey(seed)
        state, timestep = reset_fn(key)
        solution = generate_solution_fn(key)

        while not timestep.last():
            action_key, key = jax.random.split(key)
            action = binpack_optimal_policy_select_action(
                timestep.observation, solution
            )
            state, timestep = step_fn(state, action)
            assert not timestep.extras["invalid_action_taken"]
            assert not timestep.extras["invalid_ems_from_env"]
        assert_trees_are_equal(state.items_placed, solution.items_placed)
