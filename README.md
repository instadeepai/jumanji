<p align="center">
    <a href="" rel="nofollow">
        <img src="docs/img/jumanji_logo.png" />
    </a>
</p>
<div align="center">
<a rel="nofollow">
    <img src="docs/img/python-badge.svg" alt="Python" />
</a>
<a rel="nofollow">
    <img src="docs/img/pypi-badge.svg" alt="Pypi" />
</a>
<a rel="nofollow">
    <img src="docs/img/license-badge.svg" alt="License" />
</a>
<a rel="nofollow">
    <img src="docs/img/cov-badge.svg" alt="Coverage" />
</a>
<a rel="nofollow">
    <img src="docs/img/mypy-badge.svg" alt="Styling" />
</a>
<a rel="nofollow">
    <img src="docs/img/cov-style.svg" alt="Styling" />
</a>
</div>

---
[**Quickstart**](#quickstart)
| [**Installation**](#installation)
| [**Citation**](#citation)
| **Changelogs**
| **Reference docs**
| [**See Also**](#see-also)

## Welcome to the Jungle! 🌴

Jumanji is a suite of RL environments written in JAX focused on providing clean and hardware-accelerated
environments for industry-driven research. Jumanji was originally started by engineers and researchers
within the Research Team at [InstaDeep](https://www.instadeep.com/), and is now developed jointly with
the open source community.

Our goals are:

1. Provide a simple, well-test API reference for JAX-based environments.
2. Make research in Reinforcement Learning more accessible:
(i) The high-performance induced by JAX-based environments allows for more robust research, i.e. fast iteration cycle and larger scale experiments;
(ii) Writing environments in JAX reduces the complexity that made RL research cumbersome, hence removing barriers to entry.
3. Facilitate the research on Reinforcement Learning for problems in the industry and
contribute to closing the gap between research works and their industrial applications.

[//]: # (- TODO: Value proposition compared to Gymnax)
[//]: # (- TODO: even if this is not new, we should still highlight the huge performance gain by using JAX, especialluy for hard industrial problems)


## Overview

Jumanji is a high-performance library for Reinforcement Learning environments written in JAX.
It contains everything needed to connect to your favorite RL framework such as [Acme](https://github.com/deepmind/acme),
[Stable Baselines3](https://github.com/DLR-RM/stable-baselines3), [RLlib](https://docs.ray.io/en/latest/rllib/index.html).
This includes:

- 🥑 **Environment API**: core abstractions for JAX-based environments and its variations, e.g. multi-agent, turn-by-turn.
- 🕹️ **Environment Suite**: a list of Reinforcement Learning environments ranging from simple games to complex NP-hard problems.
- ⚙️ **Wrappers and Utilities**: all the bells and whistle to efficiently work with our environments.
- 🎓 **Educational Examples and User Guides**: a few guides to facilitate Jumanji's adoption and highlight the value add of JAX-based environments.

## What does Jumanji look like?

Practitioners will find Jumanji's interface familiar
as it combines the widely adopted [OpenAI Gym](https://github.com/openai/gym)
and [DeepMind Environment](https://github.com/deepmind/dm_env) interfaces.
From OpenAI Gym, we adopted the idea of a `registry` and the `render` method,
while our `TimeStep` structure gets its inspiration from `dm_env.TimeStep`.

```python
import jax
import jumanji

# Create a Jumanji environment using the registry
env = jumanji.make('Snake-6x6-v0')

# Instantiate your (jit-able) environment
key = jax.random.PRNGKey(0)
state, timestep = jax.jit(env.reset)(key)

# (Optional) Render the env state
env.render(state)

# Interact with the environment
action = env.action_spec().generate_value()          # (dummy) action selection
state, timestep = jax.jit(env.step)(state, action)   # take a step and observe the next state and time step
```
where:

- `state` (`TypeVar`) corresponds to the internal state of an environment containing all the information required
to take a step when executing an action. This shouldn't be confused with the `observation` contains in the `timestep`,
which is the one perceived by the agent.
- `timestep` (`TimeStep`) is a dataclass containing `step_type`, `reward`, `discount`, `observation`, `extras`.
This structure is similar to [`dm_env.TimeStep`](https://github.com/deepmind/dm_env/blob/master/docs/index.md)
except for the `extra` field that was added to allow users to retrieve information that are neither
part of the agent's observation nor part of the environment internal state.
A notable advantage of using TimeStep, as opposed to the Gym interface,
is how it nicely handles termination and truncation thanks to `StepType`
([discussion](https://github.com/openai/gym/issues/2510)).

Being written in JAX, Jumanji's environments benefit from many of its features including
automatic vectorization/parallelization (`jax.vmap`, `jax.pmap`) and JIT-compilation (`jax.jit`),
which can be composed arbitrarily.
We provide an example of this below, where we use `jax.vmap` and `jax.lax.scan` to generate a batch
of rollouts in the `Snake` environment.

```python
import jax

import jumanji
from jumanji.wrappers import AutoResetWrapper

env = jumanji.make("Snake-6x6-v0")  # Creates the snake environment.
env = AutoResetWrapper(env)         # Automatically reset the environment when an episode terminates.

batch_size, rollout_length = 7, 5
num_actions = env.action_spec().num_values

random_key = jax.random.PRNGKey(0)
key1, key2 = jax.random.split(random_key)

def step_fn(state, key):
  action = jax.random.randint(key=key, minval=0, maxval=num_actions, shape=())
  new_state, timestep = env.step(state, action)
  return new_state, timestep

def run_n_step(state, key, n):
  random_keys = jax.random.split(key, n)
  state, rollout = jax.lax.scan(step_fn, state, random_keys)
  return rollout

# Instantiate a batch of environment states
keys = jax.random.split(key1, batch_size)
state, timestep = jax.vmap(env.reset)(keys)

# Collect a batch of rollouts
keys = jax.random.split(key2, batch_size)
rollout = jax.vmap(run_n_step, in_axes=(0, 0, None))(state, keys, rollout_length)

# Shape and type of rollout:
# TimeStep(step_type=(7, 5), reward=(7, 5), discount=(7, 5), observation=(7, 5, 6, 6, 5), extras=None)
```

## Examples 🕹️

For a more in-depth examples of running with Jumanji environments, check out our colab notebooks:

| Example           | Topic          | Colab |
|-------------------|----------------|:-----:|
| Online Q-Learning | RL Training ([Anakin](https://arxiv.org/abs/2104.06272)) |   <a href="https://colab.research.google.com/github/instadeep/jumanji/examples/anakin_snake.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>    |


## Environments 🌍

TODO(whoever) - Add the reference to the doc (point to the documentation) and the code (class).

TODO(Clement) - need to mention the Game types we support

| Environment                     | Category      | Type         | source | Description |
|---------------------------------|---------------|--------------|--------|-------------|
| 🐍 Snake                        | game          | single-agent | code   | doc         |
| 4️⃣ Connect4                    | game          | turn-by-turn | code   | doc         |
| 📬 Travelling Salesman Problem  | combinatorial | single-agent | code   | doc         |
| 🎒 Knapsack                     | combinatorial | single-agent | code   | doc         |
| 🪢 Routing                      | combinatorial | multi-agent  | code   | doc         |
| 📦 3D BinPacking Problem        | combinatorial | single-agent | code   | doc         |


### Registry and Versioning

Similarly to OpenAI Gym, Jumanji keeps a strict versioning of its environments for reproducibility reasons.
We maintain a registry of standard environments with their configuration.
For each one, a version suffix is appended, e.g. `Snake-6x6-v0`.
When changes are made to environments that might impact learning results,
the number is increased by one to prevent potential confusion.


#### Registered configurations:

TODO: move this to a separate page containing the listing table.
We can then just import it and reuse it more easily in the doc

🐍 **Snake**

- `Snake-6x6-v0`, Snake game on a board of size `6x6`.
- `Snake-12x12-v0`, Snake game on a board of size `12x12`.

4️⃣ **Connect4**

- `Connect4-v0`, the classic [Connect4](https://en.wikipedia.org/wiki/Connect_Four) game.

📬 **Travelling Salesman Problem**

- `TSP50`, TSP problem with 50 cities (randomly generated).
- `TSP100`, TSP problem with 100 cities (randomly generated).
- `TSP150`, TSP problem with 150 cities (randomly generated).
- `TSP200`, TSP problem with 200 cities (randomly generated).

🎒 **Knapsack**

- `Knapsack50-v0`,
- `Knapsack100-v0`
- `Knapsack200-v0`
- `Knapsack250-v0`

🪢 **Routing**

- `Routing-n3-8x8-v0`, 3 routes on a `8x8` board.
- `Routing-n4-12x12-v0`, 4 routes on a `12x12` board.
- `Routing-n5-16x16-v0`, 5 routes on a `16x16` board.

📦 **3D BinPacking Problem**

- `BinPack-toy-v0`, a fixed problem instance containing 20 items to pack in a 20ft container.
- `BinPack-rand20-v0`, randomly generated instances containing 20 items.
- `BinPack-rand40-v0`, randomly generated instances containing 40 items.
- `BinPack-rand100-v0`, randomly generated instances containing 100 items.

## Installation 🎬

You can install the latest released version of Jumanji from PyPI via:
```bash
pip install jumanji
```
or you can install the latest development version from GitHub directly:
```bash
pip install git+https://github.com/instadeepai/jumanji.git
```
We tested Jumanji on Python 3.7, 3.8 and 3.9.
Note that because JAX installation is different depending on your hardware accelerator,
we advise users to explicitly install the correct JAX version,
see the [official installation guide](https://github.com/google/jax#installation).

Matplotlib is used for rendering the `BinPack` and `Snake` environments. If you want to visualise the environments you will need a GUI backend. For example on Linux you can install Tk via:
```bash
apt-get install python3-tk
```
Or with conda using:
```bash
conda install tk
```
For a list of backends you could use check out Matplotlib [backends](https://matplotlib.org/stable/users/explain/backends.html).

## Contributing 🤝
Contributions welcome! See our issue tracker for [good first issues](https://github.com/instadeepai/jumanji-internal/labels/good%20first%20issue).
Please read our [contributing guidelines](./CONTRIBUTING.md) for details on
how to submit pull requests, our Contributor License Agreement and community guidelines.

## Citing Jumanji ✏️
If you use Jumanji in your work, please cite the library using:
```
@software{jumanji2022github,
  author = {Clément Bonnet and Donal Byrne and Victor Le and Laurence Midgley
        and Daniel Luo and Cemlyn Waters and Sasha Abramowitz and Edan Toledo
        and Cyprien Courtot and Matthew Morris and Daniel Furelos Blanco
        and Thomas D. Barrett and Alexandre Laterre},
  title = {Jumanji: Industry-Driven Hardware-Accelerated RL Environments},
  url = {https://github.com/instadeepai/jumanji},
  version = {0.0.1},
  year = {2022},
}
```
In the above bibtex entry, the version number is intended to be that of the latest release,
and the year corresponds to the project's open-source release.

## See Also
Other works embraced the approach of writing RL environments in JAX.
In particular, we suggest users check the following sister repositories:

- 🦾 [Brax](https://github.com/google/brax) is a differentiable physics engine that simulates environments made up of rigid bodies, joints, and actuators.
- 🏋️‍ [Gymnax](https://github.com/RobertTLange/gymnax) implements classic environments including classic control, bsuite, MinAtar and a collection of meta RL tasks.
- 🌳 [Evojax](https://github.com/google/evojax) provides tools to enable neuroevolution algorithms to work with neural networks running across multiple TPU/GPUs.
- 🤖 [Qdax](https://github.com/adaptive-intelligent-robotics/QDax) is a library to accelerate Quality-Diversity and neuro-evolution algorithms through hardware accelerators and parallelization.

## Acknowledgements 🙏

The development of this library was supported with Cloud TPUs
from Google's [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC) 🌤️
