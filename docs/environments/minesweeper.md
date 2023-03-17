# Minesweeper Environment

<p align="center">
        <img src="../env_anim/minesweeper.gif" height="300"/>
</p>

We provide here a Jax JIT-able implementation of the game _Minesweeper_.


## Observation
The observation given to the agent consists of:
- `board`: jax array (int32) of shape `(num_rows, num_cols)`:
    each cell contains `-1` if not yet explored, or otherwise the number of mines in
    the 8 adjacent squares.
- `action_mask`: jax array (bool) of shape `(num_rows, num_cols)`:
    indicates which actions are valid (not yet explored squares). This can also be determined from
    the board which will have an entry of `-1` in all of these positions.
- `num_mines`: jax array (int32) of shape `()`, indicates the number of mines to locate.
- `step_count`: jax array (int32) of shape `()`:
    specifies how many timesteps have elapsed since environment reset.


## Action
The action space is a `MultiDiscreteArray` of integer values representing coordinates of the square
to explore, e.g. `[3, 6]` for the cell located on the third row and sixth column. If either a mined
square or an already explored square is selected, the episode terminates (the latter are termed
_invalid actions_).

Also, exploring a square will reveal only the contents of that square. This differs slightly from
the usual implementation of the game, which automatically and recursively reveals neighbouring
squares if there are no adjacent mines.


## Reward
The reward is configurable, but default to `+1` for exploring a new square that does not contain a
mine, and `0` otherwise (which also terminates the episode). The episode also terminates if the
board is solved.


## Registered Versions 📖
- `Minesweeper-v0`, the classic [game](https://en.wikipedia.org/wiki/Minesweeper) on a 10x10 grid
with 10 mines to locate.
