# PacMan Environment

<p align="center">
        <img src="../env_anim/pacman.gif" width="600"/>
</p>

We provide here a minimal Jax JIT-able implementation of the game pacman. The game is played in a 2D matrix where a cell is a free space (light blue), a wall (black), pacman (yellow) or a ghost.


The goal is for the agent (yellow) collect all of the pellets (teal) on the map without touching any of the ghosts. The agent only recieves a reward of +10 when collecting a pellet for the first time. Pellets are removed from the map after being collected.

The power-ups (purple) change the colour of the ghosts to dark blue for 60 ingame steps. When the ghosts are in this state, the player can touch them which causes them to return to the center of the map and become dormant for a fixed number of steps.

The agent selects an action at each timestep (up, left, right, down, no-op) which determines the direction they wil travel for that step. However, even if an action is in an invalid direction it will still be taken as input and the player will remain stationary. If the no-op action is used the player will not stop but instead take the last action that was selected.

The game takes place on a fixed map and the same map is generated on each reset. The generator can increase the difficulty of the game by reducing the dormant time of the ghosts, reducing their initial movement delay.

## Observation
As an observation, the agent has access to the current maze configuration in the array named
`grid`. It also has access to its current position `player_locations`, the ghosts' locations

`ghost_locations`, the power-pellet locations `power_up_location`, the time left for the scatter state `frightened_state_time`, the pellet locations `pellet_locations` and the action
mask `action_mask`.

- `agent_position`: Position(row, col) (int32) each of shape `()`, agent position in the maze.

- `ghost_locations`: jax arry (int) of shape `(4,4)`, with the (y,x) coordinates of each ghost

- `power_up_locations`: jax arry (int) of shape `(4,4)`, with the (y,x) coordinates of each power-pellet

- `pellet_locations`: jax arry (int) of shape `(4,4)`, with the (y,x) coordinates of each pellet

- `frightened_state_time`: jax array (int32) of shape `()`, number of steps left of the scatter state.

- `action_mask`: jax array (bool) of shape `(4,)`, binary values denoting whether each action is
possible.

An example 5x5 observation `grid` array, is shown below. 1 represents a wall, and 0 represents free
space.

```
[0, 1, 0, 0, 0],
[0, 1, 0, 1, 1],
[0, 1, 0, 0, 0],
[0, 0, 0, 1, 1],
[0, 0, 0, 0, 0]
```


## Action
The action space is a `DiscreteArray` of integer values in the range of [0, 4]. I.e. the agent can
take one of four actions: up (`0`), right (`1`), down (`2`), left (`3`) or no-op (`4`). If an invalid action is
taken, or an action is blocked by a wall, a no-op is performed and the agent's position remains
unchanged.


## Reward
Pacman is a dense reward setting, where the agent recieves a reward of +10 for each pellet collected. The agent also recieve a reward of 20 for collecting a power pellet. The game ends when the agent has collected all 316 pellets on the map or touches a ghost.


## Registered Versions 📖
- `PacMan-v0`, Pacman in a 31x28 map with simple grid observations.
