# Connector Environment

<p align="center">
        <img src="../env_anim/connector.gif" width="300"/>
</p>

The Connector environment contains multiple agents spawned in a grid world with each agent representing a start and end position that need to be connected. The main goal of the environment is to connect each start and end position in as few steps as possible. However, when an agent moves it leaves behind a path, which is impassable by all agents. Thus agents need to cooperate in order to allow each other to connect to their own targets without overlapping.

An episode ends when all agents have connected to their targets or no agents can make any further moves due to being blocked.

## Observation

At each step observation contains 3 items: a grid for each agent, an action mask for each agent and the episode step count.

 - `grid`: jax array (int32) of shape `(grid_size, grid_size)`, a size-configurable 2D matrix that represents pairs of points that need to be connected. The **position** of an agent has to connect to its **target**, leaving a **path** behind it as it moves across the grid forming its route. Each agent connects to only 1 target.
 - `action_mask`: jax array (bool) of shape `(num_agents,)`, indicates which actions each agent can take.
 - `step_count`: jax array (int32) of shape `()`, represents how many steps have been taken in the environment since the last reset.

Each agent is passed their own grid so the grid observation is of shape `(num_agents, grid_size, grid_size)`, similarly action masks are of shape `(num_agents, 5)`, however the step is a common value for all agents and thus is a scalar.


### Encoding
Each agent has 3 components represented in the observation space: position, target, and path. Each agent in the environment will have an integer representing their components.

 - Positions are encoded starting from 2 in multiples of 3: 2, 5, 8, …
 - Targets are encoded starting from 3 in multiples of 3: 3, 6, 9, …
 - Paths appear in the location of the head once it moves, starting from 1 in multiples of 3: 1, 4, 7, …

Every group of 3 corresponds to 1 agent: (1,2,3), (4,5,6), …

Example:
```
Agent1[path=1, position=2, target=3]
Agent2[path=4, position=5, target=6]
Agent3[path=7, position=8, target=9]
```

For example, on a 6x6 grid, the starting observation is shown below.

```
[[ 2  0  3  0  0  0]
 [ 1  0  4  4  4  0]
 [ 1  0  5  9  0  0]
 [ 1  0  0  0  0  0]
 [ 0  0  0  8  0  0]
 [ 0  0  6  7  7  7]]
```

### Current Agent (multi-agent)

Given that this is a multi-agent environment, each agent gets its own observation thus we must have a way to represent the current agent, so that the actor/learner knows which agent its actions will apply to. The current agent is always encoded as `(1,2,3)` in the observations. However this notion of current agent only exists in the observations, in the state agent 0 is always encoded as `(1,2,3)`.

The implementation shifts all other agents' values to make the `(1,2,3)` values represent the current agent, so in each agents observation it will be represented by `(1,2,3)`.
This means that the agent with the values `(4,5,6)` will always be the next agent to act.

## Action

The action space is a `MultiDiscreteArray` of shape `(num_agents,)` of integer values in the range of `[0, 4]`. Each value corresponds to an agent moving in 1 of 4 cardinal directions or taking the no-op action. [0,1,2,3,4] -> [No Op, Up, Right, Down, Left].

## Reward

The reward can be either:
- **Dense**: +1.0 for each agent that connects at that step and -0.03 for each agent that has not connected yet.

Rewards are provided in the shape `(num_agents,)` so that each agent can have a reward.



## Registered Versions 📖
