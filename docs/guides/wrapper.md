# Wrapper

The `Wrapper` interface is used for extending Jumanji `Environment` to add features like auto reset and vectorised environments.
Jumanji provides wrappers to convert a Jumanji `Environment` to a DeepMind or Gym environment, and a Brax environment into a Jumanji `Environment`.

## Brax to Jumanji
Below is an example of how to convert a brax environment into a Jumanji environment. In this example Walker2d
terminates when 1000 steps are reached.

```python
import brax.envs
import jax.random
import jumanji.wrappers

brax_env = brax.envs.create("walker2d")
env = jumanji.wrappers.BraxToJumanjiWrapper(brax_env)

key = jax.random.PRNGKey(0)
state, timestep, extra = env.reset(key)
action = jax.random.normal(key, [brax_env.action_size])
state, timestep, extra = env.step(state, action)
...
```

## Auto-reset an Environment
Below is an example of how to extend the functionality of the Snake environment to automatically reset whenever the environment reaches a terminal state.
The Snake game terminates when the snake hits the wall, using the `AutoResetWrapper` the environment will be reset once a terminal state has been reached.

```python
import jax.random

import jumanji.wrappers

env = jumanji.make("Snake6")
env = jumanji.wrappers.AutoResetWrapper(env)

key = jax.random.PRNGKey(0)
state, timestep, extra = env.reset(key)
print("New episode")
for i in range(100):
    action = env.action_spec().generate_value()  # Returns jnp.array(0) when using Snake.
    state, timestep, extra = env.step(state, action)
    if timestep.first():
        print("New episode")
```
