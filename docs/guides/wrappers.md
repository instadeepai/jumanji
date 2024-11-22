# Wrappers

The `Wrapper` interface is used for extending Jumanji `Environment` to add features like auto reset
and vectorised environments. Jumanji provides wrappers to convert a Jumanji `Environment` to a
DeepMind or Gym environment.

## Jumanji to DeepMind Environment
We can also convert our Jumanji environments to a DeepMind environment:
```python
import jumanji.wrappers

env = jumanji.make("Snake-6x6-v0")
dm_env = jumanji.wrappers.JumanjiToDMEnvWrapper(env)

timestep = dm_env.reset()
action = dm_env.action_spec.generate_value()
next_timestep = dm_env.step(action)
...
```

## Jumanji To Gymnasium
We can also convert our Jumanji environments to a [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) environment!
Below is an example of how to convert a Jumanji environment into a Gymnasium environment.

```python
import jumanji.wrappers

env = jumanji.make("Snake-6x6-v0")
gym_env = jumanji.wrappers.JumanjiToGymWrapper(env)

obs, info = gym_env.reset()
action = gym_env.action_space.sample()
observation, reward, term, trunc, info = gym_env.step(action)
...
```

## Auto-reset an Environment
Below is an example of how to extend the functionality of the Snake environment to automatically
reset whenever the environment reaches a terminal state. The Snake game terminates when the snake
hits the wall, using the `AutoResetWrapper` the environment will be reset once a terminal state has
been reached.

```python
import jax.random

import jumanji.wrappers

env = jumanji.make("Snake-6x6-v0")
env = jumanji.wrappers.AutoResetWrapper(env)

key = jax.random.PRNGKey(0)
state, timestep = env.reset(key)
print("New episode")
for i in range(100):
    action = env.action_spec.generate_value()  # Returns jnp.array(0) when using Snake.
    state, timestep = env.step(state, action)
    if timestep.first():
        print("New episode")
```
