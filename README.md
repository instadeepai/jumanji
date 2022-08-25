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
[**Installation**](#installation)
| [**Quickstart**](#what-does-jumanji-look-like)
| [**Examples**](#examples)
| [**Environments**](#environments)
| [**Citation**](#citation)
| [**See Also**](#see-also)
| [**Reference docs**](instadeepai.github.io/jumanji-internal/)

## Welcome to the Jungle! 🌴

Jumanji is a suite of Reinforcement Learning (RL) environments written in JAX focused on providing clean, hardware-accelerated
environments for industry-driven research. Jumanji is helping pioneer a new wave of hardware-accelerated research and development in the
exciting field of RL. Jumanji's high-speed environments enable faster iterations and larger-scale experimentation
while simultaneously reducing complexity and removing barriers of entry.

Jumanji was originally started by engineers and researchers
within the Research Team at [InstaDeep](https://www.instadeep.com/) and is now developed jointly with
the open-source community. We are looking forward to seeing what the future of JAX-powered RL holds and what we
can do with the RL community. Don't forget to watch 🔭 and star 🌟 the repo, so you don't miss out on any of the
exciting advances and future development of Jumanji!

### Goals 🚀

1. Provide a simple, well-test API reference for JAX-based environments.
2. Make research in RL more accessible.
3. Facilitate the research on RL for problems in the industry and
contribute to closing the gap between research and industrial applications.

### Features 📜

- 👾 **Framework Compatability**: easily connect to your favorite RL framework such as [Acme](https://github.com/deepmind/acme),
[Stable Baselines3](https://github.com/DLR-RM/stable-baselines3), and [RLlib](https://docs.ray.io/en/latest/rllib/index.html).
- 🥑 **Environment API**: core abstractions for JAX-based environments and their variations, e.g. multi-agent, turn-by-turn.
- 🕹️ **Environment Suite**: a list of RL environments ranging from simple games to complex NP-hard problems.
- ⚙️ **Wrappers and Utilities**: all the bells and whistles to efficiently work with our environments.
- 🎓 **Educational Examples and User Guides**: a few guides to facilitate Jumanji's adoption and highlight the value add of JAX-based environments.

## Installation 🎬

You can install the latest released version of Jumanji from PyPI via:
```bash
pip install jumanji
```
or you can install the latest development version from GitHub directly:
```bash
pip install git+https://github.com/instadeepai/jumanji.git
```
We tested Jumanji on Python 3.8 and 3.9.
Note that because the installation of JAX differs depending on your hardware accelerator,
we advise users to explicitly install the correct JAX version,
see the [official installation guide](https://github.com/google/jax#installation).

**Rendering:** Matplotlib is used for rendering the `BinPack` and `Snake` environments.
If you want to visualize the environments you will need a GUI backend.
For example, on Linux, you can install Tk via: `apt-get install python3-tk`, or using conda: `conda install tk`.
For a list of backends, you could use check out Matplotlib [backends](https://matplotlib.org/stable/users/explain/backends.html).

## What does Jumanji look like? 👀

Practitioners will find Jumanji's interface familiar
as it combines the widely adopted [OpenAI Gym](https://github.com/openai/gym)
and [DeepMind Environment](https://github.com/deepmind/dm_env) interfaces.
From OpenAI Gym, we adopted the idea of a `registry` and the `render` method,
while our `TimeStep` structure is inspired by `dm_env.TimeStep`.

### Basic API Usage 🧑‍💻

```python
import jax
import jumanji

# Instantiate a Jumanji environment using the registry
env = jumanji.make('Snake-6x6-v0')

# Reset your (jit-able) environment
key = jax.random.PRNGKey(0)
state, timestep = jax.jit(env.reset)(key)

# (Optional) Render the env state
env.render(state)

# Interact with the (jit-able) environment
action = env.action_spec().generate_value()          # (dummy) action selection
state, timestep = jax.jit(env.step)(state, action)   # take a step and observe the next state and time step
```
where:

- `state` corresponds to the internal state of an environment containing all the information required
to take a step when executing an action. This shouldn't be confused with the `observation` contained in the `timestep`,
which is the one perceived by the agent.
- `timestep` is a dataclass containing `step_type`, `reward`, `discount`, `observation`, and `extras`.
This structure is similar to [`dm_env.TimeStep`](https://github.com/deepmind/dm_env/blob/master/docs/index.md)
except for the `extras` field that was added to allow users to retrieve information that is neither
part of the agent's observation nor part of the environment's internal state.
A notable advantage of using `TimeStep`, as opposed to the Gym interface,
is how it nicely handles termination and truncation thanks to `StepType`
([discussion](https://github.com/openai/gym/issues/2510)).

### Advanced Usage 🧑‍🔬
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

For more in-depth examples of running with Jumanji environments, check out our colab notebooks:

| Example           | Topic          | Colab |
|-------------------|----------------|:-----:|
| Online Q-Learning | RL Training ([Anakin](https://arxiv.org/abs/2104.06272)) |   <a href="https://colab.research.google.com/github/instadeep/jumanji/examples/anakin_snake.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>    |


## Environments 🌍

Jumanji implements different types of environments ranging from simple games to NP-hard problems,
from single-agent to multi-agent and turn-by-turn games.

| Environment                                                                 | Category      | Type         | source                                                                                                           | Description                                                   |
|-----------------------------------------------------------------------------|---------------|--------------|------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| 🐍 Snake                  | game          | single-agent | [code](https://github.com/instadeepai/jumanji-internal/tree/develop/jumanji/environments/games/snake)            | [doc](http://127.0.0.1:8000//jumanji/api/environments/snake/) |
| 4️⃣  Connect4          | game          | turn-by-turn | [code](https://github.com/instadeepai/jumanji-internal/tree/develop/jumanji/environments/games/connect4)         | [doc](http://127.0.0.1:8000//jumanji/api/environments/connect4/)                                                           |
| 📬 Travelling Salesman Problem | combinatorial | single-agent | [code](https://github.com/instadeepai/jumanji-internal/tree/develop/jumanji/environments/combinatorial/tsp)      | [doc](http://127.0.0.1:8000//jumanji/api/environments/tsp/)                                                           |
| 🎒 Knapsack        | combinatorial | single-agent | [code](https://github.com/instadeepai/jumanji-internal/tree/develop/jumanji/environments/combinatorial/knapsack) | [doc](http://127.0.0.1:8000//jumanji/api/environments/knapsack/)                                                           |
| 🪢 Routing        | combinatorial | multi-agent  | [code](https://github.com/instadeepai/jumanji-internal/tree/develop/jumanji/environments/combinatorial/routing)  | [doc](http://127.0.0.1:8000//jumanji/api/environments/routing/)                                                           |
| 📦 3D BinPacking Problem | combinatorial | single-agent | [code](https://github.com/instadeepai/jumanji-internal/tree/develop/jumanji/environments/combinatorial/binpack)  | [doc](http://127.0.0.1:8000//jumanji/api/environments/binpack/)                                                           |


### Registry and Versioning 📖

Similarly to OpenAI Gym, Jumanji keeps a strict versioning of its environments for reproducibility reasons.
We maintain a registry of standard environments with their configuration.
For each one, a version suffix is appended, e.g. `Snake-6x6-v0`.
When changes are made to environments that might impact learning results,
the version number is increased by one to prevent potential confusion. For a full list of registered versions
of each environment, check out the environments' documentation.

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
        and Cyprien Courtot and Matthew Morris and Daniel Furelos-Blanco
        and Nathan Grinsztajn and Thomas D. Barrett and Alexandre Laterre},
  title = {Jumanji: Industry-Driven Hardware-Accelerated RL Environments},
  url = {https://github.com/instadeepai/jumanji},
  version = {0.1.0},
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
