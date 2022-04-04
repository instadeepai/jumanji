# Connect 4 Environment

We provide here a Jax JIT-able implementation of the Connect 4 environment.

The environment contains two agents competing against each other, playing turn by turn.

The goal of each agent is to win by connecting 4 tokens horizontally, vertically or diagonally.
If the board gets full before any of the two agents connects 4, it's a draw and the
reward is 0 for both players.
If an agent makes an invalid move, they lose instantly. The reward for loosing is -1, while the one
for winning is 1.

The board is 6 rows by 7 columns.
The observation is a dataclass containing a view of the board from the point of view of the current
player and a mask indicating valid actions.
