from jumanji.snake import Snake
from validation import EnvironmentLoop, RandomAgent


def run_snake_random_jit() -> None:
    """Runs a random agent in Snake using the jitted EnvironmentLoop. This serves as an
    example of how to use an agent on an Environment using the EnvironmentLoop."""
    snake_env = Snake()
    random_agent = RandomAgent(action_spec=snake_env.action_spec())
    environment_loop = EnvironmentLoop(
        environment=snake_env,
        agent=random_agent,
        n_steps=20,
        batch_size=10,
    )
    environment_loop.run(num_steps=1_000, ms=True)


if __name__ == "__main__":
    run_snake_random_jit()
