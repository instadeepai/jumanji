from jumanji import specs
from jumanji.pcb_grid import PcbGridEnv
from jumanji.wrappers import MultiToSingleEnvironment
from validation import EnvironmentLoop, RandomAgent


def run_pcb_random_jit() -> None:
    """Runs a random agent in pcb grid using the jitted EnvironmentLoop. This serves as an
    example of how to use an agent on an Environment using the EnvironmentLoop."""
    pcb_env = MultiToSingleEnvironment(PcbGridEnv())
    action_spec: specs.BoundedArray = pcb_env.action_spec()  # type: ignore
    random_agent = RandomAgent(action_spec=action_spec)
    environment_loop = EnvironmentLoop(
        environment=pcb_env,
        agent=random_agent,
        n_steps=20,
        batch_size=10,
    )
    environment_loop.run(num_steps=1_000, ms=True)


if __name__ == "__main__":
    run_pcb_random_jit()
