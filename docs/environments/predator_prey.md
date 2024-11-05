# Predator-Prey Flock Environment

[//]: # (TODO: Add animated plot)

Environment modelling two competing flocks/swarms of agents:

- Predator agents are rewarded for contacting prey agents, or for proximity to prey agents.
- Prey agents are conversely penalised for being contacted by, or for proximity to predators.

Each set of agents can consist of multiple agents, each independently
updated, and with their own independent observations. The agents occupy a square
space with periodic boundary conditions. Agents have a limited view range, i.e. they
only partially observe their local environment (and the locations of neighbouring agents within
range). Rewards are also assigned individually to each agent dependent on their local state.

## Observation

Each agent generates an independent observation, an array of values
representing the distance along a ray from the agent to the nearest neighbour, with
each cell representing a ray angle (with `num_vision` rays evenly distributed over the agents
field of vision). Prey and prey agent types are visualised independently to allow agents
to observe both local position and type.

- `predators`: jax array (float) of shape `(num_predators, 2 * num_vision)` in the unit interval.
- `prey`: jax array (float) of shape `(num_prey, 2 * num_vision)` in the unit interval.

## Action

Agents can update their velocity each step by rotating and accelerating/decelerating. Values
are clipped to the range `[-1, 1]` and then scaled by max rotation and acceleration
parameters. Agents are restricted to velocities within a fixed range of speeds.

- `predators`: jax array (float) of shape (num_predators, 2) each corresponding to `[rotation, acceleration]`.
- `prey`: jax array (float) of shape (num_prey, 2) each corresponding to `[rotation, acceleration]`.

## Reward

Rewards are generated for each agent individually. They are generally dependent on proximity, so
their scale can depend on agent density and interaction ranges.

- `predators`: jax array (float) of shape `(num_predators,)`, individual predator agent rewards.
- `prey`: jax array (float) of shape `(num_prey,)`, individual prey rewards.
