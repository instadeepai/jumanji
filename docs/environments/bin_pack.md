# BinPack Environment

<p align="center">
        <img src="../env_anim/bin_pack.gif" width="500"/>
</p>

We provide here an implementation of the 3D [bin packing problem](https://en.wikipedia.org/wiki/Bin_packing_problem).
In this problem, the goal of the agent is to efficiently pack a set of boxes (items) of different
sizes into a single container with as little empty space as possible. Since there is only 1 bin,
this formulation is equivalent to the 3D-knapsack problem.


## Observation
The observation given to the agent provides information on the available empty space (called EMSs),
the items that still need to be packed, and information on what actions are valid at this point.
The full observation is as follows:

- `ems`: `EMS` tree of jax arrays (float if `normalize_dimensions` else int32) each of shape
    `(obs_num_ems,)`, coordinates of all EMSs at the current timestep.

- `ems_mask`: jax array (bool) of shape `(obs_num_ems,)`, indicates the EMSs that are valid.

- `items`: `Item` tree of jax arrays (float if `normalize_dimensions` else int32) each of shape
    `(max_num_items,)`, characteristics of all items for this instance.

- `items_mask`: jax array (bool) of shape `(max_num_items,)`, indicates the items that are valid.

- `items_placed`: jax array (bool) of shape `(max_num_items,)`, indicates the items that have been
    placed so far.

- `action_mask`: jax array (bool) of shape `(obs_num_ems, max_num_items)`, mask of the joint action
    space: `True` if the action `[ems_id, item_id]` is valid.


## Action
The action space is a `MultiDiscreteArray` of 2 integer values representing the ID of an EMS
(space) and the ID of an item. For instance, `[1, 5]` will place item 5 in EMS 1.


## Reward
The reward could be either:

- **Dense**: normalized volume (relative to the container volume) of the item packed by taking
    the chosen action. The computed reward is equivalent to the increase in volume utilization
    of the container due to packing the chosen item. If the action is invalid, the reward is 0.0
    instead.

- **Sparse**: computed only at the end of the episode (otherwise, returns 0.0). Returns the volume
    utilization of the container (between 0.0 and 1.0). If the action is invalid, the action is
    ignored and the reward is still returned as the current container utilization.


## Registered Versions 📖
- `BinPack-v2`, 3D bin-packing problem with a solvable random generator that generates up to 20
items maximum, that can handle 40 EMSs maximum that are given in the observation.
