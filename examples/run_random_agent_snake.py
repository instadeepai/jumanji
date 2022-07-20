from jumanji.snake import Snake
from validation import JaxEnvironmentLoop, RandomAgent


def run_snake_random_jit() -> None:
    """Runs a random agent in Snake using the jitted Jax Environment Loop. This serves as an
    example of how to use an agent on a JaxEnv environment using the JaxEnvironmentLoop."""
    snake_env = Snake()
    random_agent = RandomAgent(action_spec=snake_env.action_spec())
    environment_loop = JaxEnvironmentLoop(
        environment=snake_env,
        agent=random_agent,
        n_steps=20,
        batch_size=10,
    )
    environment_loop.run(num_steps=1_000, ms=True)


if __name__ == "__main__":
    run_snake_random_jit()
