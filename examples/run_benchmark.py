import logging
from typing import List, Optional, Type, Union

import hydra
from omegaconf import DictConfig, OmegaConf

from jumanji.jax import JaxEnv
from jumanji.jax.wrappers import DeepMindEnvWrapper
from jumanji.utils import (
    DeepMindEnvBenchmarkLoop,
    JaxEnvBenchmarkLoop,
    JaxEnvironmentLoop,
)

ENV_LOOP_CLASSES = {
    "JaxEnvBenchmarkLoop": JaxEnvBenchmarkLoop,
    "DeepMindEnvBenchmarkLoop": DeepMindEnvBenchmarkLoop,
    "JaxEnvironmentLoop": JaxEnvironmentLoop,
}
WRAPPERS = {"DeepMindEnvWrapper": DeepMindEnvWrapper}


def run_benchmark(
    env: JaxEnv,
    env_loop_cls: Union[
        Type[JaxEnvBenchmarkLoop],
        Type[DeepMindEnvBenchmarkLoop],
        Type[JaxEnvironmentLoop],
    ],
    env_wrappers: List = None,
    num_episodes: Optional[int] = None,
    num_env_steps: Optional[int] = None,
    n_steps: int = 1,
    batch_size: int = 1,
    ms: bool = False,
    print_: bool = True,
) -> None:
    """Runs an environment loop on an provided environment for a certain number of steps or
    episodes. This is for benchmark purposes, to test the speed of different environments,
    with different environment loops.

    Args:
        env: JaxEnv environment.
        env_loop_cls: type of environment loop to use. Can be either of:
            - JaxEnvBenchmarkLoop
            - DmEnvBenchmarkLoop
            - JaxEnvironmentLoop
        env_wrappers: list of environment wrappers.
        num_episodes: number of episodes to play in the environment.
        num_env_steps: number of steps to take in the environment (either num_episodes or
            num_env_steps should be None).
        n_steps: number of steps to run in a sequence for the JaxEnvironmentLoop.
        batch_size: number of states to run in parallel for the JaxEnvironmentLoop.
        ms: True to have time displayed in milliseconds, False to have it in seconds.
        print_: True to print, False to not print but log with info level.

    """
    if env_wrappers:
        for wrapper in env_wrappers:
            env = wrapper(env)
    if issubclass(env_loop_cls, JaxEnvironmentLoop):
        env_loop: JaxEnvironmentLoop = env_loop_cls(
            env, n_steps=n_steps, batch_size=batch_size
        )
    elif issubclass(env_loop_cls, JaxEnvBenchmarkLoop) or issubclass(
        env_loop_cls, DeepMindEnvBenchmarkLoop
    ):
        env_loop: Union[  # type: ignore
            JaxEnvBenchmarkLoop, DeepMindEnvBenchmarkLoop
        ] = env_loop_cls(env)
    else:
        raise NotImplementedError
    env_loop.run(
        num_episodes=num_episodes, num_steps=num_env_steps, ms=ms, print_=print_
    )


@hydra.main(config_path="configs", config_name="config.yaml")
def run(cfg: DictConfig) -> None:
    """Loads benchmark config and run a benchmark.

    Args:
        cfg: hydra DictConfig.

    """
    logging.info("Configs:\n{}".format(OmegaConf.to_yaml(cfg)))
    run_benchmark(
        env=hydra.utils.instantiate(cfg.environment),
        env_loop_cls=ENV_LOOP_CLASSES[cfg.environment_loop.cls],
        env_wrappers=[
            WRAPPERS[wrapper_name] for wrapper_name in cfg.environment_loop.wrappers
        ],
        num_episodes=cfg.num_episodes,
        num_env_steps=cfg.num_env_steps,
        n_steps=cfg.environment_loop.n_steps,
        batch_size=cfg.environment_loop.batch_size,
        ms=cfg.print_time_in_ms,
        print_=cfg.print_time,
    )


if __name__ == "__main__":
    # Change configs in configs/config.yaml.
    run()
