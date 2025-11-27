import time
from functools import partial

import jax
import jax.numpy as jnp
import pytest

import jumanji
from jumanji.environments.routing.connector.generator import UniformRandomGenerator


# TODO: remove static arg once in wrapper as should be able to access
# self._env._generator._grid_size
@partial(jax.jit, static_argnames=("grid_size",))
def raycast(
    grid: jax.Array, pos: tuple[int, int], dir: tuple[int, int], grid_size: int
) -> tuple[jax.Array, jax.Array]:
    """
    Raycasts from a starting position in a specific direction until a non-zero
    value is found or the boundaries of the grid are reached.

    Args:
        grid: 2D JAX array (H, W).
        start_pos: Tuple or array (row, col) integer index to start from.
        direction: Tuple or array (dr, dc) integer direction.

    Returns:
        The first non-zero value encountered, 0 if the ray exits the grid without hitting anything.
    """
    start_r, start_c = pos
    dr, dc = dir

    # Generate steps starting from 1 (assuming the start position is not checked)
    steps = jnp.arange(1, grid_size + 1)  # [1, 2, 3, ..., max_len]

    # Generate the full path of indices
    path_r = start_r + steps * dr
    path_c = start_c + steps * dc

    # Sample all values along the generated path using advanced JAX indexing.
    # The key optimization here is using 'mode="fill"' and 'fill_value=0'.
    # Any index that is out of bounds (OOB) is automatically set to 0.
    sampled_vals = grid.at[path_r, path_c].get(
        wrap_negative_indices=False,  # negative inds are OOB
        indices_are_sorted=True,  # Faster indexing
        unique_indices=True,  # faster indexing
        mode="fill",  # fills if OOB with `fill_value`
        fill_value=0,  # if OOB treat as empty cell
    )

    # Find the index of the first non-zero value in the sampled path.
    # Since OOB locations were filled with 0, this automatically finds the first
    # non-zero value that is also within the grid boundaries.
    # If no non-zero value exists, argmax returns 0.
    first_hit_index = jnp.argmax(sampled_vals != 0)
    hit_value = sampled_vals[first_hit_index]
    # If no hit then return 0 distance.
    # If hit then increase distance by 1 otherwise neighbours would be 0 distance
    distance = first_hit_index + (hit_value != 0)
    return hit_value, distance


def speed_test():
    n_envs = 64
    n_agents = 15
    grid_size = 23
    env = jumanji.make("Connector-v3", generator=UniformRandomGenerator(grid_size, n_agents))

    # Create a batch of grids.
    key = jax.random.key(0)
    keys = jax.random.split(key, n_envs)
    states, _ = jax.vmap(env.reset)(keys)
    grids = states.grid

    # Define all directions
    directions = jnp.array(
        [
            [0, 1],  # Right
            [0, -1],  # Left
            [1, 0],  # Down
            [-1, 0],  # Up
            [1, 1],  # Down-Right
            [-1, -1],  # Up-Left
            [1, -1],  # Down-Left
            [-1, 1],  # Up-Right
        ]
    )

    # Define all positions on the grid
    grid_size = int(env._generator._grid_size)
    positions = jax.random.randint(key, (n_agents, 2), 0, grid_size)

    # Create a large nested vmap as per user request:
    # 1. vmap over positions (outermost).
    # 2. vmap over directions.
    # 3. vmap over grids (innermost).
    vmapped_raycast = jax.vmap(  # over positions
        jax.vmap(  # over directions
            jax.vmap(raycast, in_axes=(0, None, None, None)),  # over grids
            in_axes=(None, None, 0, None),
        ),
        in_axes=(None, 0, None, None),
    )

    print("Running large nested vmap for speed test...")

    # JIT compile and run the vmapped function
    print(f"{type(grids)=} | {type(positions)=} | {type(directions)=} | {type(grid_size)=}")
    vals, dists = vmapped_raycast(grids, positions, directions, grid_size)  # compile
    vals.block_until_ready()
    start = time.perf_counter()
    vals, dists = vmapped_raycast(grids, positions, directions, grid_size)
    vals.block_until_ready()

    total_time = time.perf_counter() - start
    print(f"Test finished. Result shape: {total_time}")


# --- Test Setup ---
@pytest.fixture
def center_target_grid():
    """
    A 5x5 grid with a target (5) at the edges and center empty.
    Used for testing rays starting from center (2,2) going OUT.

    [[5, 0, 1, 0, 6],
     [0, 0, 0, 0, 0],
     [3, 0, 0, 0, 4],
     [0, 0, 0, 0, 0],
     [7, 0, 2, 0, 8]]
    """
    grid = jnp.zeros((5, 5), dtype=int)
    # Cardinals
    grid = grid.at[0, 2].set(1)  # Up
    grid = grid.at[4, 2].set(2)  # Down
    grid = grid.at[2, 0].set(3)  # Left
    grid = grid.at[2, 4].set(4)  # Right
    # Diagonals
    grid = grid.at[0, 0].set(5)  # Up-Left
    grid = grid.at[0, 4].set(6)  # Up-Right
    grid = grid.at[4, 0].set(7)  # Down-Left
    grid = grid.at[4, 4].set(8)  # Down-Right
    return grid


@pytest.fixture
def occlusion_grid():
    """
    Grid to test that ray stops at first hit.
    Row 2: [0, 10, 20, 30, 0]
    """
    grid = jnp.zeros((5, 5), dtype=int)
    grid = grid.at[2, 1].set(10)
    grid = grid.at[2, 2].set(20)
    grid = grid.at[2, 3].set(30)
    return grid


# --- Tests ---
@pytest.mark.parametrize(
    "direction, expected_val",
    [
        ((-1, 0), 1),  # Up
        ((1, 0), 2),  # Down
        ((0, -1), 3),  # Left
        ((0, 1), 4),  # Right
    ],
)
def test_cardinal_directions(center_target_grid, direction, expected_val):
    """Test standard up/down/left/right hits from center."""
    start_pos = (2, 2)  # Center
    # Start (2,2) -> Edge is 2 steps away (e.g., 2,2 -> 1,2 -> 0,2)
    expected_dist = 2

    result, dist = raycast(center_target_grid, start_pos, direction, grid_size=5)

    assert result == expected_val
    assert dist == expected_dist


@pytest.mark.parametrize(
    "direction, expected_val",
    [
        ((-1, -1), 5),  # Up-Left
        ((-1, 1), 6),  # Up-Right
        ((1, -1), 7),  # Down-Left
        ((1, 1), 8),  # Down-Right
    ],
)
def test_diagonal_directions(center_target_grid, direction, expected_val):
    """Test all 4 diagonal hits from center."""
    start_pos = (2, 2)  # Center
    # Start (2,2) -> Corner is 2 steps away (e.g., 2,2 -> 1,1 -> 0,0)
    expected_dist = 2

    result, dist = raycast(center_target_grid, start_pos, direction, grid_size=5)

    assert result == expected_val
    assert dist == expected_dist


def test_no_hit_returns_zero(center_target_grid):
    """Test that casting into the void returns 0."""
    # From center (2,2), moving like a knight (2, 1) hits nothing in this grid
    start_pos = (2, 2)
    direction = (2, 1)
    result, _ = raycast(center_target_grid, start_pos, direction, grid_size=5)
    assert result == 0


def test_occlusion(occlusion_grid):
    """
    Test that the ray picks the CLOSEST object and ignores objects behind it.
    Grid Row 2: [0, 10, 20, 30, 0]
    """
    # Cast Right from (2, 0) -> Should hit 10 at dist 1
    assert raycast(occlusion_grid, (2, 0), (0, 1), 5) == (10, 1)

    # Cast Right from (2, 1) -> Should hit 20 at dist 1
    assert raycast(occlusion_grid, (2, 1), (0, 1), 5) == (20, 1)

    # Cast Left from (2, 4) -> Should hit 30 at dist 1
    assert raycast(occlusion_grid, (2, 4), (0, -1), 5) == (30, 1)


def test_immediate_neighbor():
    """Test hitting an object immediately adjacent to start."""
    grid = jnp.zeros((3, 3), dtype=int)
    grid = grid.at[0, 1].set(99)
    # Start at [0,0], cast Right (0,1) -> Should hit [0,1]
    result, dist = raycast(grid, (0, 0), (0, 1), 3)
    assert result == 99
    assert dist == 1


# --- Boundary & Wrap-Around Tests ---


def test_negative_index_wrap_prevention():
    """
    CRITICAL TEST: JAX/Numpy normally wraps negative indices (index -1 is the last item).
    In a grid, moving Left from col 0 results in col -1.
    We MUST ensure this counts as 'Out of Bounds' (0), not 'Right side of grid'.
    """
    grid = jnp.zeros((5, 5), dtype=int)

    # Place a value on the far right edge
    grid = grid.at[2, 4].set(99)

    # Start at the far left edge [2, 0]
    # Cast LEFT (-1). Index becomes [2, -1].
    # If wrapping occurs, it would hit [2, 4] (val 99).
    # If correctly masked, it returns 0.
    start_pos = (2, 0)
    direction = (0, -1)

    result, dist = raycast(grid, start_pos, direction, grid_size=5)
    assert result == 0, "Ray wrapped around negative index (Left Edge)!"
    assert dist == 0


def test_negative_row_wrap_prevention():
    """Same as above, but for Top Edge wrapping to Bottom Edge."""
    grid = jnp.zeros((5, 5), dtype=int)
    grid = grid.at[4, 2].set(99)  # Bottom edge

    # Start Top edge [0, 2], cast UP (-1, 0)
    start_pos = (0, 2)
    direction = (-1, 0)

    result, dist = raycast(grid, start_pos, direction, grid_size=5)
    assert result == 0, "Ray wrapped around negative index (Top Edge)!"
    assert dist == 0


def test_positive_boundary_overflow():
    """Test that going off the positive edge (Right/Bottom) simply returns 0."""
    grid = jnp.zeros((5, 5), dtype=int)
    grid = grid.at[2, 0].set(99)  # Far Left

    # Start Far Right [2, 4], cast Right (0, 1) -> Indices [2, 5], [2, 6]...
    # Should not crash, should return 0
    result, dist = raycast(grid, (2, 4), (0, 1), 5)
    assert result == 0
    assert dist == 0


def test_diagonal_boundary_overflow(center_target_grid):
    """
    Test that diagonal rays going off the grid do not wrap around.
    Using center_target_grid which has values in all corners.
    """
    grid_size = 5

    # 1. Start Top-Left (0,0), value 5. Go Up-Left (-1, -1).
    # If wrapped, it would hit Bottom-Right (4,4) which is 8.
    result, dist = raycast(center_target_grid, (0, 0), (-1, -1), grid_size)
    assert result == 0, "Diagonal Up-Left wrap detected!"
    assert dist == 0

    # 2. Start Bottom-Right (4,4), value 8. Go Down-Right (1, 1).
    # If wrapped, it would hit Top-Left (0,0) which is 5.
    result, dist = raycast(center_target_grid, (4, 4), (1, 1), grid_size)
    assert result == 0, "Diagonal Down-Right overflow detected!"
    assert dist == 0

    # 3. Start Top-Right (0,4), value 6. Go Up-Right (-1, 1).
    # If wrapped, it would hit Bottom-Left (4,0) which is 7.
    result, dist = raycast(center_target_grid, (0, 4), (-1, 1), grid_size)
    assert result == 0, "Diagonal Up-Right wrap detected!"
    assert dist == 0


if __name__ == "__main__":
    speed_test()
