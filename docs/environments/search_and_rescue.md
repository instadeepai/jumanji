# 🚁 Search & Rescue

[//]: # (TODO: Add animated plot)

Multi-agent environment, modelling a group of agents searching the environment
for multiple targets. Agents are individually rewarded for finding a target
that has not previously been detected.

Each agent visualises a local region around it, creating a simple segmented view
of locations of other agents in the vicinity. The environment is updated in the
following sequence:

- The velocity of searching agents are updated, and consequently their positions.
- The positions of targets are updated.
- Agents are rewarded for being within a fixed range of targets, and the target
  being within its view cone.
- Targets within detection range and an agents view cone are marked as found.
- Local views of the environment are generated for each search agent.

The agents are allotted a fixed number of steps to locate the targets. The search
space is a uniform space with unit dimensions, and wrapped at the boundaries.

## Observations

- `searcher_views`: jax array (float) of shape `(num_searchers, num_vision)`. Each agent
  generates an independent observation, an array of values representing the distance
  along a ray from the agent to the nearest neighbour, with  each cell representing a
  ray angle (with `num_vision` rays evenly distributed over the agents field of vision).
  For example if an agent sees another agent straight ahead and `num_vision = 5` then
  the observation array could be

  ```
  [1.0, 1.0, 0.5, 1.0, 1.0]
  ```

  where `1.0` indicates there is no agents along that ray, and `0.5` is the normalised
  distance to the other agent.
- `target_remaining`: float in the range [0, 1]. The normalised number of targets
  remaining to be detected (i.e. 1.0 when no targets have been found).
- `time_remaining`: float in the range [0, 1]. The normalised number of steps remaining
  to locate the targets (i.e. 0.0 at the end of the episode).

## Actions

Jax array (float) of `(num_searchers, 2)` in the range [-1, 1]. Each entry in the
array represents an update of each agents velocity in the next step. Searching agents
update their velocity each step by  rotating and accelerating/decelerating. Values
are clipped to the range `[-1, 1]` and then scaled by max rotation and acceleration
parameters. Agents are restricted to velocities within a fixed range of speeds.

## Rewards

Jax array (float) of `(num_searchers, 2)`. Rewards are generated for each agent individually.
Agents are rewarded 1.0 for locating a target that has not already been detected.
