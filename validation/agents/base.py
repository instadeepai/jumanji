import abc
from typing import Dict, Tuple

from chex import Array, PRNGKey

from jumanji.jax.types import Action, Extra
from validation.types import TrainingState, Transition


class Agent(abc.ABC):
    """Abstraction for implementing agents."""

    @abc.abstractmethod
    def init_training_state(self, key: PRNGKey) -> TrainingState:
        """Initializes the learning state of the agent.

        Args:
            key: random key used for the initialization of the learning state.

        Returns:
            training state containing the parameters, optimizer state and counter.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def select_action(
        self,
        training_state: TrainingState,
        observation: Array,
        key: PRNGKey,
        extra: Extra = None,
    ) -> Action:
        """Select an action accounting for the current observation.

        Args:
            training_state: contains the parameters, optimizer state and counter.
            observation: jax array.
            key: random key to sample an action.
            extra: optional field of type Extra containing additional information
                about the environment.

        Returns:
            action.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def sgd_step(
        self, training_state: TrainingState, batch_traj: Transition
    ) -> Tuple[TrainingState, Dict]:
        """Computes an RL learning step.

        Args:
            training_state: contains the parameters, optimizer state and counter.
            batch_traj: batch of trajectories in the environment to compute the loss function.

        Returns:
            new training state after update.
            metrics containing information about the training, loss, etc.

        """
        raise NotImplementedError
