from jumanji import specs
from jumanji.pcb_grid import PcbGridEnv
from jumanji.wrappers import MultiToSingleJaxEnv
from validation import JaxEnvironmentLoop, RandomAgent


def run_pcb_random_jit() -> None:
    """Runs a random agent in pcb grid using the jitted Jax Environment Loop. This serves as an
    example of how to use an agent on a JaxEnv environment using the JaxEnvironmentLoop."""
    pcb_env = MultiToSingleJaxEnv(PcbGridEnv())
    action_spec: specs.BoundedArray = pcb_env.action_spec()  # type: ignore
    random_agent = RandomAgent(action_spec=action_spec)
    environment_loop = JaxEnvironmentLoop(
        environment=pcb_env,
        agent=random_agent,
        n_steps=20,
        batch_size=10,
    )
    environment_loop.run(num_steps=1_000, ms=True)


if __name__ == "__main__":
    run_pcb_random_jit()
