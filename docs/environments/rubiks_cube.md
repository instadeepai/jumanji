# Rubik's Cube Environment

<p align="center">
        <img src="../env_anim/rubiks_cube.gif" height="200"/>
</p>

We provide here a Jax JIT-able implementation of the _Rubik's cube_. The environment contains an implementation of the classic 3x3x3 cube by default, and configurably other sizes. The goal of the agent is to match all stickers on each face to a single colour. On resetting the environment the cube will be randomly scrambled with a configurable number of turns (by default 100).

## Observation
The observation given to the agent gives a view of the current state of the cube,
- `cube`: jax array (int8) of shape `(6, cube_size, cube_size)` whose values are in `[0, 1, 2, 3, 4, 5]`
  (corresponding to the different sticker colors). The indices of the array specify the sticker position - first
the face (in the order _up_, _front_, _right_, _back_, _left_, _down_) and then the row and column.
Note that the orientation of each face is as follows:
  - UP: LEFT face on the left and BACK face pointing up
  - FRONT: LEFT face on the left and UP face pointing up
  - RIGHT: FRONT face on the left and UP face pointing up
  - BACK: RIGHT face on the left and UP face pointing up
  - LEFT: BACK face on the left and UP face pointing up
  - DOWN: LEFT face on the left and FRONT face pointing up
- `step_count`: jax array (int32) of shape `()`, representing the number of steps in the episode thus far.

## Action
The action space is a `MultiDiscreteArray`, specifically a tuple of an index between 0 and 5 (since there
are 6 faces), an index between 0 and `cube_size//2` (the number of possible depths), and an index between
0 and 2 (3 possible directions). An action thus consists of three pieces of information:
- Face to turn,
- Depth of the turn (possible depths are between `0` representing the outer layer and `cube_size//2`
representing the layer closest to the middle),
- Direction of turn (possible directions are clockwise, anti-clockwise, or a half turn).



## Reward
The reward function is configurable, but by default is the fully sparse reward giving `+1` for solving the cube and otherwise `0`.
The episode terminates if either the cube is solved or a configurable horizon (by default `200`) is reached.

## Registered Versions 📖
- `RubiksCube-v0`, the standard Rubik's Cube [puzzle](https://en.wikipedia.org/wiki/Rubik%27s_Cube)
with faces of size 3x3.
- `RubiksCube-partly-scrambled-v0`, an easier version of the standard Rubik's Cube puzzle with faces
of size 3x3 yet only 7 scrambles at reset time, making it technically maximum 7 actions away from
the solution.
