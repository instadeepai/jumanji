# Snake Environment 🐍

<p align="center">
        <img src="../env_anim/snake.gif" width="500"/>
</p>

We provide here an implementation of the _Snake_ environment from
[(Bonnet et al., 2021)](https://arxiv.org/abs/2111.00206). The goal of the agent is to navigate
in a grid world (by default of size 12x12) to collect as many fruits as possible without colliding
with its own body (i.e. looping on itself).


## Observation

- `grid`: jax array (float) of shape `(num_rows, num_cols, 5)`, feature maps (image) that include
    information about the fruit, the snake head, its body and tail.

- `step_count`: jax array (int32) of shape `()`, current number of steps in the episode.

- `action_mask`: jax array (bool) of shape `(4,)`, array specifying which directions the snake can
    move in from its current position.


## Action
The action space is a `DiscreteArray` of integer values: `[0,1,2,3]` -> `[Up, Right, Down, Left]`.


## Reward
The reward is `+1` upon collection of a fruit and `0` otherwise.


## Registered Versions 📖
- `Snake-v1`: Snake game on a board of size 12x12 with a time limit of `4000`.
