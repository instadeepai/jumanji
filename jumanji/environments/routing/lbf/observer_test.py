import jax.numpy as jnp

from jumanji.environments.routing.lbf.observer import LbfGridObserver, LbfVectorObserver
from jumanji.environments.routing.lbf.types import Food, State

# Levels:
# agent grid
# [1, 2, 0],
# [2, 0, 1],
# [0, 0, 0],

# food grid
# [0, 0, 0],
# [0, 4, 0],
# [3, 0, 0],

# IDs:
# agent grid
# [a0, a1, 0],
# [a2, 0, a3],
# [0, 0, 0],

# food grid
# [0, 0, 0],
# [0, f0, 0],
# [f1, 0, 0],


def test_grid_observer(state: State) -> None:
    observer = LbfGridObserver(fov=1, grid_size=3)
    obs = observer.state_to_observation(state)
    expected_agent_0_view = jnp.array(
        [
            [
                # other agent levels
                [-1, -1, -1],
                [-1, 1, 2],
                [-1, 2, 0],
            ],
            [
                # food levels
                [-1, -1, -1],
                [-1, 0, 0],
                [-1, 0, 4],
            ],
            [
                # access (where can the agent go?)
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
        ]
    )

    assert jnp.all(obs.agents_view[0, ...] == expected_agent_0_view)
    assert jnp.all(
        obs.action_mask[0, ...] == jnp.array([True, False, False, False, False, True])
    )

    expected_agent_1_view = jnp.array(
        [
            [
                [-1, -1, -1],
                [1, 2, 0],
                [2, 0, 1],
            ],
            [
                [-1, -1, -1],
                [0, 0, 0],
                [0, 4, 0],
            ],
            [
                [0, 0, 0],
                [0, 1, 1],
                [0, 0, 0],
            ],
        ]
    )
    assert jnp.all(obs.agents_view[1, ...] == expected_agent_1_view)
    assert jnp.all(
        obs.action_mask[1, ...] == jnp.array([True, False, True, False, False, True])
    )

    expected_agent_3_view = jnp.array(
        [
            [
                [2, 0, -1],
                [0, 1, -1],
                [0, 0, -1],
            ],
            [
                [0, 0, -1],
                [4, 0, -1],
                [0, 0, -1],
            ],
            [
                [0, 1, 0],
                [0, 1, 0],
                [1, 1, 0],
            ],
        ]
    )

    assert jnp.all(obs.agents_view[3, ...] == expected_agent_3_view)
    assert jnp.all(
        obs.action_mask[3, ...] == jnp.array([True, True, False, True, False, True])
    )

    # test different fov
    observer = LbfGridObserver(fov=3, grid_size=3)
    # test eaten food is not visible
    eaten = jnp.array([True, False])
    foods = Food(
        eaten=eaten,
        id=state.foods.id,
        position=state.foods.position,
        level=state.foods.level,
    )
    state = state.replace(foods=foods)

    obs = observer.state_to_observation(state)
    expected_agent_2_view = jnp.array(
        [
            [
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, 1, 2, 0, -1],
                [-1, -1, -1, 2, 0, 1, -1],
                [-1, -1, -1, 0, 0, 0, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ],
            [
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, 0, 0, 0, -1],
                [-1, -1, -1, 0, 0, 0, -1],
                [-1, -1, -1, 3, 0, 0, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
        ]
    )
    assert jnp.all(obs.agents_view[2, ...] == expected_agent_2_view)
    assert jnp.all(
        obs.action_mask[2, ...] == jnp.array([True, False, True, False, False, True])
    )


def test_vector_observer(state: State) -> None:
    observer = LbfVectorObserver(fov=1, grid_size=3)
    obs = observer.state_to_observation(state)
    expected_agent_0_view = jnp.array(
        [1, 1, 4, -1, -1, 0, 0, 0, 1, 0, 1, 2, 1, 0, 2, -1, -1, 0]
    )
    assert jnp.all(obs.agents_view[0, ...] == expected_agent_0_view)
    assert jnp.all(
        obs.action_mask[0, ...] == jnp.array([True, False, False, False, False, True])
    )

    expected_agent_2_view = jnp.array(
        [1, 1, 4, 2, 0, 3, 1, 0, 2, 0, 0, 1, 0, 1, 2, -1, -1, 0]
    )
    assert jnp.all(obs.agents_view[2, ...] == expected_agent_2_view)
    assert jnp.all(
        obs.action_mask[2, ...] == jnp.array([True, False, False, False, False, True])
    )

    # test different fov
    observer = LbfVectorObserver(fov=3, grid_size=3)
    # test eaten food is not visible
    eaten = jnp.array([True, False])
    foods = Food(
        eaten=eaten,
        id=state.foods.id,
        position=state.foods.position,
        level=state.foods.level,
    )
    state = state.replace(foods=foods)

    obs = observer.state_to_observation(state)
    expected_agent_3_view = jnp.array(
        [-1, -1, 0, 2, 0, 3, 1, 2, 1, 0, 0, 1, 0, 1, 2, 1, 0, 2]
    )
    assert jnp.all(obs.agents_view[3, ...] == expected_agent_3_view)
    assert jnp.all(
        obs.action_mask[3, ...] == jnp.array([True, True, False, True, True, True])
    )
