import jax.numpy as jnp

from jumanji.snake.types import Position


def position_from_coordinates(coordinates: jnp.ndarray) -> Position:
    """Transforms an array of coordinates into a Position.

    Args:
        coordinates: array of coordinates of shape (*, 2).

    Returns:
        Position with row and col of shape (*,).
    """
    return Position(*(x.squeeze() for x in jnp.split(coordinates, 2, axis=-1)))
