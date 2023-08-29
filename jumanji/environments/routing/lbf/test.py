import time

import jax
import jax.numpy as jnp

# key = jax.random.PRNGKey(0)
# keys = jax.random.split(key, 1000000)
#
#
# env = LevelBasedForaging()
# reset = jax.jit(env.reset)
# step = jax.jit(env.step)
#
# state, ts = reset(key)
# # print(ts.observation)
# state, ts = step(state, jnp.array([1, 2, 3, 4]))
# print(ts.observation)
import jumanji

env = jumanji.make("LevelBasedForaging-v0")
env.step
print("h")
