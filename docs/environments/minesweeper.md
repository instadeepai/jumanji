# Minesweeper Environment

<p align="center">
        <img src="../env_anim/minesweeper.gif" height="300"/>
</p>

We provide here a Jax JIT-able implementation of the game _Minesweeper_.

## Observation
The observation given to the agent consists of:
- `board`: jax array (int32) of shape `(board_height, board_width)`, representing a grid whose
entries are `-1` if the square has not yet been explored, or otherwise the number of surrounding squares
which have mines (between 0 and 8).
- `action_mask`: jax array (bool) of shape `(board_height, board_width)`, indicating which squares have not yet been explored
(this can also be determined from the board which will have an entry of `-1` in all of these positions).
- `num_mines`: jax array (int32) of shape `()`, representing the number of mines to locate.
- `step_count`: jax array (int32) of shape `()`, representing the number of steps in the episode thus far.

## Action
The action space is a `MultiDiscreteArray` of integer values representing coordinates of the square to explore.
If either a mined square or an already explored square is selected, the episode terminates (the latter are termed "invalid actions").

Also, exploring a square will reveal only the contents of that square. This differs slightly from the usual
implementation of the game, which automatically and recursively reveals neighbouring squares if there
are no adjacent mines.

## Reward
The reward is configurable, but default to `+1` for exploring a new square that does not contain a mine, and `0`
otherwise (which also terminates the episode). The episode also terminates if the board is solved.

## Registered Versions 📖
- `Minesweeper-v0`, the classic [game](https://en.wikipedia.org/wiki/Minesweeper).
