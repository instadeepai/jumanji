import jax.numpy as jnp

from jumanji.snake.types import Position
from jumanji.snake.utils import position_from_coordinates


def test_position_from_coordinates() -> None:
    """Validates position_from_coordinates method from utils.
    Checks the conversion to a single Position and to an array of positions.
    """
    coordinates = jnp.array([3, 2])
    position = position_from_coordinates(coordinates)
    assert position == Position(jnp.array(3), jnp.array(2))
    coordinates = jnp.array([[3, 2], [1, 4], [0, 1]])
    position = position_from_coordinates(coordinates)
    assert position == Position(jnp.array([3, 1, 0]), jnp.array([2, 4, 1]))
