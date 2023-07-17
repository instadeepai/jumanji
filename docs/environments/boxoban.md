# Boxoban Environment 👾

<p align="center">
        <img src="../env_anim/boxoban.gif" width="500"/>
</p>

We provide here a Jax implementation of the _Boxoban_ environment inspired by the popular Sokoban puzzle game. This implementation utilizes the dataset from <a href="#ref1">[1]</a> and the graphical resources from the <a href="#ref2">[2]</a> which also implements Boxoban and a range of variants of Sokoban. The goal of the agent is to navigate a grid world.

## Observation

- `grid`: Array (uint8) of shape `(10, 10, 2)`, representing the variable grid and fixed grid respectively. The variable grid contains encodings of objects that can move (boxes and the agent). The fixed grid contains encodings for fixed objects (walls and targets).
- `step_count`: Array (int32) of shape `()`, current number of steps in the episode.

## Object Encodings

| Object       | Encoding |
|--------------|----------|
| Empty Space  | 0        |
| Wall         | 1        |
| Target       | 2        |
| Agent        | 3        |
| Box          | 4        |

## Action
The action space is an Array (int32) with possible values of `[0,1,2,3]` -> `[Up, Down, Left, Right]`.

## Reward
The reward function is

- `-0.1` every step taken in the environment.
- `+1` for each box moved onto a target location and `-1` for each box moved off a target location.
- `+10` once all 4 boxes are placed on their targets

## Episode Termination

The episode terminates under the following conditions:

- The step limit of 120 is reached.
- The problem is solved (all 4 boxes are placed on targets).

## Graphics

| Type             | Graphic                                                           |
|------------------|-------------------------------------------------------------------|
| Wall             | ![Wall](../img/boxoban_img/wall.png)                  |
| Floor            | ![Floor](../img/boxoban_img/floor.png)               |
| Target    | ![BoxTarget](../img/boxoban_img/box_target.png)    |
| Box on Target    | ![BoxTarget](../img/boxoban_img/box_on_target.png) |
| Box Off Target   | ![BoxOffTarget](../img/boxoban_img/box.png)       |
| Agent Off Target | ![PlayerOffTarget](../img/boxoban_img/agent.png)  |
| Agent On Target  | ![PlayerOnTarget](../img/boxoban_img/agent_on_target.png) |

## Registered Versions 📖
- `Boxoban-v0`: Boxoban game using DeepMind dataset.

## References
<a id="ref1">[1]</a> Guez, A., Mirza, M., Gregor, K., Kabra, R., Racaniere, S., Weber, T., Raposo, D., Santoro, A., Orseau, L., Eccles, T., Wayne, G., Silver, D., Lillicrap, T., Valdes, V. (2018). An investigation of Model-free planning: boxoban levels. Available at [https://github.com/deepmind/boxoban-levels](https://github.com/deepmind/boxoban-levels)

<a id="ref2">[2]</a> Schrader, M. (2018). Gym-sokoban. Available at [https://github.com/mpSchrader/gym-sokoban](https://github.com/mpSchrader/gym-sokoban)
