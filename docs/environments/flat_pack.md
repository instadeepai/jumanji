# FlatPack Environment

<p align="center">
        <img src="../env_anim/flat_pack.gif" width="500"/>
</p>

We provide here a Jax JIT-able implementation of a simple _jigsaw_ puzzle. The goal of the agent is to place
all the jigsaw pieces in the correct locations on an empty 2D puzzle board. Each time an episode resets a
new puzzle and set of piece is created. Pieces are randomly shuffled and rotated.

## Observation
The observation given to the agent gives a view of the current state of the puzzle as well as
all pieces that can be placed.

- `current_board`: jax array (float32) of shape `(num_rows, num_cols)` with values in the range
    `[0, num_pieces]` (corresponding to the number of each piece). This board will have zeros
    where no pieces have been placed and numbers corresponding to each piece where that particular
    pieces has been paced.

- `pieces`: jax array (float32) of shape `(num_pieces, 3, 3)` of all possible pieces in the
    current puzzle. These pieces are shuffled and rotated. Pieces will always have shape `(3, 3)`.

- `action_mask`: jax array (bool) of shape `(num_pieces, 4, num_rows-2, num_cols-2)`, representing
    which actions are possible given the current state of the board. The first index indicates the
    number of pieces in a given puzzle. The second index indicates the number of times a piece may be rotated.
    The third and fourth indices indicate the row and column coordinate of where a piece may be placed respectively.
    These values will always be `num_rows-2` and `num_cols-2` respectively to make it impossible for an agent to
    place a piece outside the current board.


## Action
The action space is a `MultiDiscreteArray`, specifically a tuple of an index between 0 and `num_pieces`,
an index between 0 and 4 (since there are 4 possible rotations), an index between 0 and `num_rows-2`
(the possible row coordinates for placing a piece) and an index between 0 and `num_cols-2`
(the possible column coordinates for placing a piece). An action thus consists of four pieces of
information:

- Piece to place,

- Number of rotations to make to a chosen piece ({0, 90, 180, 270} degrees),

- Row coordinate for placing the rotated piece,

- Column coordinate for placed the rotated piece.


## Reward
The reward function is configurable, but by default is a fully dense reward giving `+1` for
each cell of a placed piece that overlaps with its correct position on the solved board. The episode
terminates if either the puzzle is solved or `num_pieces` steps have been taken by an agent.


## Registered Versions 📖
- `FlatPack-v0`, a jigsaw puzzle with 7 rows and 7 columns containing 3 row pieces and 3 column pieces
    for a total of 9 pieces in the puzzle. This version has a dense reward.
