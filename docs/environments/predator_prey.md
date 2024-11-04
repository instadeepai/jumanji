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

Rewards can be either sparse or proximity-based.

### Sparse

- `predators`: jax array (float) of shape `(num_predators,)`, predators are rewarded a fixed amount
  for coming into contact with a prey agent. If they are in contact with multiple prey, only the
  nearest agent is selected.
- `prey`: jax array (float) of shape `(num_predators,)`, prey are penalised a fix negative amount if
  they come into contact with a predator agent.

### Proximity

- `predators`: jax array (float) of shape `(num_predators,)`, predators are rewarded with an amount
  scaled linearly with the distance to the prey agents, summed over agents in range.
- `prey`: jax array (float) of shape `(num_predators,)`, prey are penalised by an amount scaled linearly
  with distance from predator agents, summed over all predators in range.
