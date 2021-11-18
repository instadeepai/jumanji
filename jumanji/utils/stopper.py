from typing import Optional


def should_terminate(
    episode_count: int,
    step_count: int,
    num_episodes: Optional[int],
    num_steps: Optional[int],
) -> bool:
    """Checks whether the training should stop depending on the number of
    episodes or steps run in the environment.

    Args:
        episode_count: current number of episodes run in the environment.
        step_count: current number of steps taken in the environment.
        num_episodes: number of episodes to play in the environment.
        num_steps: number of steps to take in the environment (either num_episodes or
            num_steps should be None).

    Returns:
        True if training should stop, else False.
    """
    return (num_episodes is not None and episode_count >= num_episodes) or (
        num_steps is not None and step_count >= num_steps
    )
