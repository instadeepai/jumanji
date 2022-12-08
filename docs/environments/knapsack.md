# Knapskack Environment

<p align="center">
        <img src="../img/knapsack.png" width="350"/>
</p>

We provide here a Jax JIT-able implementation of the [knapskack problem](https://en.wikipedia.org/wiki/Knapsack_problem).

The knapsack problem is a famous problem in combinatorial optimization. The goal is to determine, given
a set of items, each with a weight and a value,
which items to include in a collection so that the total weight is less than or equal
to a given limit and the total value is as large as possible.

The decision problem form of the knapsack problem is NP-complete, thus there is no known
algorithm both correct and fast (polynomial-time) in all cases.

When the environment is reset, a new problem instance is generated, by sampling weights and values
from a uniform distribution between 0 and 1. The weight limit of the knapsack is a parameter of the
environment.
A trajectory terminates when no further item can be added to the knapsack, or the last action
is invalid.

## Observation
The observation given to the agent provides information on the total problem, the items already added
to the knapsack and environment meta info such as number of steps taken and the remaining budget.

**Observation Spec**:

- `problem` jax array (float32) of shape `(num_items, 2)`, shows an array of weights/values of the items to be packed into the knapsack.
- `first_item`: jax array (int32), gives the index of the last item added.
- `last_item`: jax array (int32), gives the index of the first item added.
- `action_mask`: jax array (bool) of shape `(num_items,)`, array of binary values denoting which items can be packed into the knapsack.

## Action
Action space is a `DiscreteArray` of integer values in the range of [0, num_items-1]. An action is the index of the
item to pack next.

```
action: 2  # Problem consisting of 4 items, choosing item 2 (among items 0, 1, 2, 3).
```

## Reward

The reward is 0 at every step, except at the last timestep when the reward is the
total value of the knapsack.

## Registered Versions 📖
- `Knapsack50-v0`
- `Knapsack100-v0`
- `Knapsack200-v0`
- `Knapsack250-v0`
