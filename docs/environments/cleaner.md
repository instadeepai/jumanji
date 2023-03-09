# Cleaner Environment

<p align="center">
        <img src="../env_anim/cleaner.gif" height="600"/>
</p>

We provide here a JAX jit-able implementation of the [Multi-Agent Cleaning](https://github.com/Bigpig4396/Multi-Agent-Reinforcement-Learning-Environment#multi-agent-cleaner)
environment.

In this environment, multiple agents must cooperatively clean the floor of a room with complex indoor barriers (black).
At the beginning of an episode, the whole floor is dirty (green).
Every time an agent (red) visits a dirty tile, it is cleaned (white).

The goal is to clean as many tiles as possible in a given time budget.

A new maze is randomly generated using a recursive division method for each new episode. Agents always start in the top left corner of the maze.

## Observation

The **observation** seen by the agent is a `NamedTuple` containing the following:

- `grid`: jax array (int) of shape `(height, width)`, array representing the grid, each tile is either dirty (0),
  clean (1), or a wall (2).
- `agents_locations`: jax array (int) of shape `(num_agents, 2)`, array specifying the x and y coordinates of every agent.
- `action_mask`: jax array (bool) of shape `(num_agents, 4)`, array specifying, for each agent, which action
  (up, right, down, left) is legal.
- `step_count`: jax array (int32) of shape `()`, number of steps elapsed in the current episode.

## Action

The action space is a `MultiDiscreteArray` containing an integer value in `[0, 1, 2, 3]` for each agent.
Each agent can take one of four actions: up (`0`), right (`1`), down (`2`), or left (`3`).

The episode terminates if any agent meets one of the following conditions:

- An invalid action is taken, or
- An action is blocked by a wall.

In both cases, the agent's position remains unchanged.

## Reward

The reward is global and shared among the agents. It is equal to the number of tiles which were cleaned during the time step, minus a penalty (0.5 by default) to encourage agents to clean the maze faster.

## Registered Versions 📖

- `Cleaner-v0`, a room of shape 10 by 10 with 3 agents.
