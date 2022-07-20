import logging
from typing import Callable, List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

from jumanji import JaxEnv
from jumanji.wrappers import JaxEnvToDeepMindEnv
from validation.benchmark_loops import (
    BenchmarkLoop,
    DeepMindEnvBenchmarkLoop,
    JaxEnvBenchmarkLoop,
)

ENV_LOOP_CLASSES = {
    "JaxEnvBenchmarkLoop": JaxEnvBenchmarkLoop,
    "DeepMindEnvBenchmarkLoop": DeepMindEnvBenchmarkLoop,
}
WRAPPERS = {
    "JaxEnvToDeepMindEnv": JaxEnvToDeepMindEnv,
}


def run_benchmark(
    env: JaxEnv,
    env_loop_cls: Callable[[JaxEnv], BenchmarkLoop],
    env_wrappers: Optional[List[Callable[[JaxEnv], JaxEnv]]] = None,
    num_episodes: Optional[int] = None,
    num_env_steps: Optional[int] = None,
    ms: bool = False,
) -> None:
    """Runs an environment loop on an provided environment for a certain number of steps or
    episodes. This is for benchmark purposes, to test the speed of different environments,
    with different environment loops.

    Args:
        env: JaxEnv environment.
        env_loop_cls: type of environment loop to use. Can be either of:
            - JaxEnvBenchmarkLoop
            - DmEnvBenchmarkLoop
        env_wrappers: list of environment wrappers.
        num_episodes: number of episodes to play in the environment.
        num_env_steps: number of steps to take in the environment (either num_episodes or
            num_env_steps should be None).
        ms: True to have time displayed in milliseconds, False to have it in seconds.

    """
    if env_wrappers:
        for wrapper in env_wrappers:
            env = wrapper(env)
    if issubclass(
        env_loop_cls, (JaxEnvBenchmarkLoop, DeepMindEnvBenchmarkLoop)  # type: ignore
    ):
        env_loop = env_loop_cls(env)
    else:
        raise NotImplementedError
    env_loop.run(num_episodes=num_episodes, num_steps=num_env_steps, ms=ms)


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
            WRAPPERS[wrapper_name] for wrapper_name in cfg.environment_loop.wrappers  # type: ignore
        ],
        num_episodes=cfg.num_episodes,
        num_env_steps=cfg.num_env_steps,
        ms=cfg.print_time_in_ms,
    )


if __name__ == "__main__":
    # Change configs in configs/config.yaml.
    run()
