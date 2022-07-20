import brax.envs

from jumanji.wrappers import BraxEnvToJaxEnv
from validation import JaxEnvironmentLoop, RandomAgent


def run_brax_halfcheetah_random_jit() -> None:
    """Runs a random agent in a Brax environment (halfcheetah) using the jitted Jax Environment
    Loop. This serves as an example of how to run an agent on a Brax environment wrapped as a JaxEnv
    environment using the JaxEnvironmentLoop."""
    brax_env = brax.envs.create(
        env_name="halfcheetah",
        episode_length=1000,
        auto_reset=False,
    )
    wrapped_env = BraxEnvToJaxEnv(brax_env)
    random_agent = RandomAgent(action_spec=wrapped_env.action_spec())
    environment_loop = JaxEnvironmentLoop(
        environment=wrapped_env,
        agent=random_agent,
        n_steps=10,
        batch_size=30,
    )
    environment_loop.run(num_steps=3_000, ms=True)


if __name__ == "__main__":
    run_brax_halfcheetah_random_jit()
