from typing import Dict, Tuple

import numpy as np
from chex import Array, PRNGKey
from jax import random

from jumanji import specs
from jumanji.types import Action, Extra
from validation.agents.base import Agent
from validation.types import TrainingState, Transition


class RandomAgent(Agent):
    """Random agent that does not learn and select actions randomly following a uniform distribution
    over the bounded action space (accepts both discrete and continuous)."""

    def __init__(self, action_spec: specs.BoundedArray):
        if not isinstance(action_spec, specs.BoundedArray):
            raise TypeError(
                f"`action_spec` must be a specs.BoundedArray (continuous or discrete), got "
                f"{action_spec} of type {action_spec.dtype}."
            )
        self._action_spec = action_spec

    def init_training_state(self, key: PRNGKey) -> TrainingState:
        """Initializes the learning state of the agent.

        Args:
            key: random key used for the initialization of the learning state.

        Returns:
            training state containing the parameters, optimizer state and counter.

        """
        training_state = TrainingState()
        return training_state

    def select_action(
        self,
        training_state: TrainingState,
        observation: Array,
        key: PRNGKey,
        extra: Extra = None,
    ) -> Action:
        """Select a random action for a specs.BoundedArray action space. Works for both discrete
        and continuous action spaces (under the condition that it is bounded).

        Args:
            training_state: contains the parameters, optimizer state and counter.
            observation: jax array.
            key: random key to sample an action.
            extra: optional field of type Extra containing additional information
                about the environment.

        Returns:
            action randomly selected in the discrete interval given by specs.BoundedArray.

        """
        if np.issubdtype(self._action_spec.dtype, np.int32) or np.issubdtype(
            self._action_spec.dtype, np.int64
        ):
            action = random.randint(
                key=key,
                shape=self._action_spec.shape,
                minval=self._action_spec.minimum,
                maxval=self._action_spec.maximum + 1,
                dtype=self._action_spec.dtype,
            )
        elif np.issubdtype(self._action_spec.dtype, np.float32) or np.issubdtype(
            self._action_spec.dtype, np.float64
        ):
            action = random.uniform(
                key=key,
                shape=self._action_spec.shape,
                dtype=self._action_spec.dtype,
                minval=self._action_spec.minimum,
                maxval=self._action_spec.maximum,
            )
        else:
            raise ValueError(
                f"`action_spec.dtype` must be integral or float, got "
                f"{self._action_spec.dtype}."
            )
        return action

    def sgd_step(
        self, training_state: TrainingState, batch_traj: Transition
    ) -> Tuple[TrainingState, Dict]:
        """There is no learning in this agent, hence this returns the same training state.

        Args:
            training_state: contains the parameters, optimizer state and counter.
            batch_traj: batch of trajectories in the environment to compute the loss function.

        Returns:
            initial training state
            metrics containing information about the training, loss, etc.

        """
        metrics: Dict = {}
        return training_state, metrics
