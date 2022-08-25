# Snake Environment 🐍

We provide here an implementation of the Snake environment from [(Bonnet et al., 2021)](https://arxiv.org/abs/2111.00206).

The goal of the agent is to navigate in a grid world (default: `(n_rows=12, n_cols=12)`)
to collect as many fruits as possible, without colliding with its own body, i.e.
looping on itself. The reward is 1 upon collection of each fruit, else 0. Its
length grows by 1 with each fruit it gathers, making it harder to survive as
the episode progresses. An episode ends if the snake exits the board,
hits itself, or after a certain number of steps (default: `time_limit=5000`).

As an observation, the agent has access to the concatenation of 5 feature maps
as channels stacked (HWC format) in an image representing the snake body,
its head, its tail, where the fruit is, as well as the order in which the cells
are organised.

<p align="center">
        <img src="../img/_snake_obs.png" width="1000"/>
</p>

## Registered Versions 📖
- `Snake-6x6-v0`, Snake game on a board of size `6x6`.
- `Snake-12x12-v0`, Snake game on a board of size `12x12`.
