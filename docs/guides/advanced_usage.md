# Advanced Usage 🧑‍🔬
Being written in JAX, Jumanji's environments benefit from many of its features including
automatic vectorization/parallelization (`jax.vmap`, `jax.pmap`) and JIT-compilation (`jax.jit`),
which can be composed arbitrarily.
We provide an example of this below, where we use `jax.vmap` and `jax.lax.scan` to generate a batch
of rollouts in the `Snake` environment.

```python
import jax

import jumanji
from jumanji.wrappers import AutoResetWrapper

env = jumanji.make("Snake-v1")  # Create a Snake environment
env = AutoResetWrapper(env)     # Automatically reset the environment when an episode terminates

batch_size = 7
rollout_length = 5
num_actions = env.action_spec().num_values

random_key = jax.random.PRNGKey(0)
key1, key2 = jax.random.split(random_key)

def step_fn(state, key):
  action = jax.random.randint(key=key, minval=0, maxval=num_actions, shape=())
  new_state, timestep = env.step(state, action)
  return new_state, timestep

def run_n_steps(state, key, n):
  random_keys = jax.random.split(key, n)
  state, rollout = jax.lax.scan(step_fn, state, random_keys)
  return rollout

# Instantiate a batch of environment states
keys = jax.random.split(key1, batch_size)
state, timestep = jax.vmap(env.reset)(keys)

# Collect a batch of rollouts
keys = jax.random.split(key2, batch_size)
rollout = jax.vmap(run_n_steps, in_axes=(0, 0, None))(state, keys, rollout_length)

# Shape and type of given rollout:
# TimeStep(step_type=(7, 5), reward=(7, 5), discount=(7, 5), observation=(7, 5, 6, 6, 5), extras=None)
```
