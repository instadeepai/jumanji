# Routing Environment

We provide here an implementation of the Routing environment using JAX.

The Routing environment contains multiple agents spawn in a grid world (default: `(rows=12, cols=12)`) with each agent representing a route that needs to
be connected to its own unique target/pin. The main goal of the environment is to connect each route and to do so optimally.
Agents need to cooperate in order to allow each other to connect to their own targets.

The rewards given to each agent can be set but by default are as follows:

For each agent:
- The reward per timestep when not connected is -0.03
- The reward for an agent connecting to its target is 0.1
- The reward for an agent blocking itself is -0.1

An episode is regarded as complete when all agents have connected to their targets/pins
or no agents can make any further moves due to being blocked.

![Routing Example](../../docs/img/routing_rollout.gif)

## Observation Space

The grid is a configurable size 2D matrix that represents pairs of points that need to be connected. Like in the Routing environment, the “head” of an agent has to connect to the “target”, leaving a “trace” behind it as it moves across the environment. Each head connects to only 1 target.

For example, on a 6x6 grid, the starting observation is shown below.

```java
[[ 0  0  0  0  0  0]
 [ 0  7 10  0  0  0]
 [ 0  0  0  6  3  0]
 [ 0  0  0  0  0  0]
 [ 0  9  0  0  0  0]
 [ 4  0  0  0  0  0]]
```


### Encoding

Heads are encoded starting from 4 in multiples of 3: 4, 7, 10, …

Targets are encoded starting from 3 in multiples of 3: 3, 6, 9, …

Traces appear in the location of the head once it moves, starting from 2 in multiples of 3: 2, 5, 8, …

Every group of 3 corresponds to 1 route: (2,3,4), (5,6,7), …

### Current Agent

The current agent is always encoded as (2,3,4).

The implementation shifts all other agents values to make the (2,3,4) values represent the current agent. This means that each subsequent agent ID will always be the previous agent id's values shifted up by 3.

## Action Space

Each route can step in 4 cardinal directions, or take no action.
