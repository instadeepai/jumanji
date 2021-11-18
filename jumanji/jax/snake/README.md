# Snake Environment 🐍

We provide here an implementation of the Snake environment from [(Bonnet et al., 2021)](https://arxiv.org/abs/2111.00206).

The goal of the agent is to navigate in a grid world (default: `(n_rows=12, n_cols=12)`)
to collect as many fruits as possible, without colliding with its own body, i.e.
looping on itself. The reward is 1 upon collection of each fruit, else 0. Its
length grows by 1 with each fruit it gathers, making it harder to survive as
the episode progresses. An episode ends if the snake exits the board,
hits itself, or after a certain number of steps (default: `time_limit=5000`).

As an observation, the agent has access to the concatenation of 5 feature maps
as channels stacked up in an image representing the snake body,
its head, its tail, where the fruit is, as well as the order in which the cells
are organised.

![Snake observation](../../../docs/img/_snake_obs.png)


# Benchmark

### Speed (Steps/s)

Speed test on 1 CPU, no GPU used.

| Environment type \ board<br>shape (n_rows, n_cols) | (5, 5)   | (10, 10) | (20, 20) |
|----------------------------------------------------|----------|----------|----------|
| DeepMindEnvWrapper (jit one step)                  | 3.10^3   | 3.10^3   | 3.10^3   |
| JaxEnv (jit one step)                              | 3-4.10^3 | 3-4.10^3 | 3-4.10^3 |
| JaxEnvironmentLoop (jit `n=20` steps)              | 5.10^5   | 3.10^5   | 1.10^5   |
