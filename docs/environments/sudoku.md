# Sudoku Environment

<p align="center">
        <img src="../env_anim/sudoku.gif" width="500"/>
</p>

We provide here a Jax JIT-able implementation of the
[Sudoku](https://en.wikipedia.org/wiki/Sudoku) puzzle game.


## Observation
The observation given to the agent consists of:


- `board`: jax array (int32) of shape (9,9):
    empty cells are represented by -1, and filled cells are represented by 0-8.
- `action_mask`: jax array (bool) of shape (9,9,9):
    indicates which actions are valid.



## Action
The action space is a `MultiDiscreteArray` of integer values representing coordinates of the square
to explore and the digits to write in the cell, e.g. `[3, 6, 8]` for writing the digit `9` in
the cell located on the fourth row and seventh column.


## Reward
The reward is `1` at the end of the episode if the board is correctly solved, and `0` in every
other case.

## Termination
An episode terminates where there is no more legal actions available, this could happen if the
board is solved or if the agent finds himself in a dead-end.

## Registered Versions 📖
- `Sudoku-v0`, the classic [game](https://en.wikipedia.org/wiki/Sudoku) on
a 9x9 grid, 10000 random puzzles with mixed difficulty are included by default.
- `Sudoku-very-easy-v0`, the classic [game](https://en.wikipedia.org/wiki/Sudoku) on
a 9x9 grid, only 1000 very-easy random puzzles (>46 clues) included by default.


## Using custom puzzle instances
If one wants to include its own database of puzzles, the `DatabaseGenerator` can be initialized with any collection of puzzles using the argument `custom_boards`.
Some references for databases of puzzle of various difficulties:
 - https://www.kaggle.com/datasets/rohanrao/sudoku
 - https://www.kaggle.com/datasets/informoney/4-million-sudoku-puzzles-easytohard


## Difficulty level as a function of number of clues
<a href="https://www.researchgate.net/figure/Number-of-clues-for-each-difficulty-level_tbl1_259525699"><img width=400 src="https://www.researchgate.net/profile/Arnab-Maji-2/publication/259525699/figure/tbl1/AS:651874038460436@1532430290710/Number-of-clues-for-each-difficulty-level.png" alt="Number of clues for each difficulty level"/></a>
Adapted from [An Algorithm for Generating only Desired Permutations for Solving Sudoku Puzzle](https://www.researchgate.net/figure/Number-of-clues-for-each-difficulty-level_tbl1_259525699).
