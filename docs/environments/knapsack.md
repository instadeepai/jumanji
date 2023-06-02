# Knapskack Environment

<p align="center">
        <img src="../env_anim/knapsack.gif" width="450"/>
</p>

We provide here a Jax JIT-able implementation of the
[knapskack problem](https://en.wikipedia.org/wiki/Knapsack_problem).

The knapsack problem is a famous problem in combinatorial optimization. The goal is to determine,
given a set of items, each with a weight and a value, which items to include in a collection so that
the total weight is less than or equal to a given limit and the total value is as large as possible.

The decision problem form of the knapsack problem is NP-complete, thus there is no known
algorithm both correct and fast (polynomial-time) in all cases.

When the environment is reset, a new problem instance is generated, by sampling weights and values
from a uniform distribution between 0 and 1. The weight limit of the knapsack is a parameter of the
environment.
A trajectory terminates when no further item can be added to the knapsack or the chosen action
is invalid.


## Observation
The observation given to the agent provides information regarding the weights and the values of all the items,
as well as, which items have been packed into the knapsack.

- `weights`: jax array (float) of shape `(num_items,)`, array of weights of the items to be
packed into the knapsack.

- `values`: jax array (float) of shape `(num_items,)`, array of values of the items to be packed
into the knapsack.

- `packed_items`: jax array (bool) of shape `(num_items,)`, array of binary values denoting which
items are already packed into the knapsack.

- `action_mask`: jax array (bool) of shape `(num_items,)`, array of binary values denoting which
items can be packed into the knapsack.


## Action
The action space is a `DiscreteArray` of integer values in the range of `[0, num_items-1]`. An
action is the index of the next item to pack.


## Reward
The reward can be either:

- **Dense**: the value of the item to pack at the current timestep.

- **Sparse**: the sum of the values of the items packed in the bag at the end of the episode.

In both cases, the reward is 0 if the action is invalid, i.e. an item that was previously
selected is selected again or has a weight larger than the bag capacity.


## Registered Versions 📖
- `Knapsack-v2`: Knapsack problem with 50 randomly generated items, a total budget of 12.5 and a
dense reward function.
