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
Welcome to Jumanji!

Jumanji is an RL environment library written in Jax focused on providing clean and fast hardware accelerated
environments for industry-driven research. Jumanji is easy to use and can be
run out of the box with other libraries such as RLlib, StableBaselines, ACME, gym and dm-env.


## Environments
| Environment                       | Type                                    | Observation                                                                                                                                                                                                                                                                                                                                                                                                    | Action                                                                                                 |
|-----------------------------------|-----------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------|
| Snake                             | Game                                    | Array (float) of shape (n_rows, n_cols, 5)                                                                                                                                                                                                                                                                                                                                                                     | array (int) `[0,1,2,3] -> [Up, Right, Down, Left]`                                                     |
| Connect4                          | Game                                    | Observation:<<br/>&nbsp;&nbsp;&nbsp;&nbsp;- board: array (int8) of shape (6, 7)<br/>&nbsp;&nbsp;&nbsp;&nbsp;- action_mask: array (int8) of shape(6, 7)                                                                                                                                                                                                                                                                                                          | array (int) `[0,1,2,3,4,5,6] one per column`                                                           |
| TSP (Travelling Salesman Problem) | Combinatorial Optimization              | Observation:<<br/>&nbsp;&nbsp;&nbsp;&nbsp;- problem: array (float32) of shape (problem_size, 2)<br/>&nbsp;&nbsp;&nbsp;&nbsp;- start_position: int32<br/>&nbsp;&nbsp;&nbsp;&nbsp;- position: int32<br/>- action_mask: array (int8) of shape (problem_size,)                                                                                                                                                                                                                              | array (int) `[0,...,problem_size-1] -> city id`                                                        |
| Knapsack                          | Combinatorial Optimization              | Observation:<br/>&nbsp;&nbsp;&nbsp;&nbsp;- problem: array (float32) of shape (problem_size, 2)<br/>&nbsp;&nbsp;&nbsp;&nbsp;- first_item: array (int32)<br/>&nbsp;&nbsp;&nbsp;&nbsp;- last_item: array (int32)<br/>&nbsp;&nbsp;&nbsp;&nbsp;- invalid_mask: array (int8) of shape (problem_size,)                                                                                                                                                                                                                | array (int) `[0,...,problem_size-1] -> item id`                                                        |
| Routing                           | Combinatorial Optimization, Multi Agent | Array (int) of shape (1, 12, 12)                                                                                                                                                                                                                                                                                                                                                                               | array (int) of shape (n_agents,) `[0,1,2,3,4] -> [No Op, Left, Up, Right, Down] per route`             |
| BinPack                           | Combinatorial Optimization              | Observation: <br/>&nbsp;&nbsp;&nbsp;&nbsp;- ems: (Empty Max Space) dataclass containing arrays of shape (6, 7)<br/>&nbsp;&nbsp;&nbsp;&nbsp;- ems_mask: array (bool) of shape (6, 7)&nbsp;&nbsp;&nbsp;&nbsp;- items: Item dataclass of arrays (float) of shape (max_num_items,)<br/>&nbsp;&nbsp;&nbsp;&nbsp;- items_mask: jax array (bool) of shape (max_num_items,)<br/>&nbsp;&nbsp;&nbsp;&nbsp;- items_placed: array (bool) of shape (max_num_items,)<br/>&nbsp;&nbsp;&nbsp;&nbsp;- action_mask: array (bool) of shape (obs_num_ems, max_num_items) | array (int) of shape (2,) <br/>`[[0,...,obs_num_ems-1], [0,...,max_num_items-1]] -> [ems_id, item_id]` |


## Quick Start

```python
import jax
import jumanji

# Create a Jumanji environment
env = jumanji.make('Snake-6x6-v0')

# Instantiate your (jit-able) environment
key = jax.random.PRNGKey(0)
state, timestep = jax.jit(env.reset)(key)

# (Optional) Render the env state
env.render(state)

# Interact with the environment
action = env.action_spec().generate_value()                 # (dummy) action selection
state, timestep = jax.jit(env.step)(state, action)   # take a step and observe the next state and time step
```

## Examples
<a href="https://colab.research.google.com/github/instadeep/jumanji/examples/anakin_snake.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
```#TODO Update with actual link from github```
For a more in-depth example of running with Jumanji environments, check out our colab notebook that
goes through running beating snake with Online Q-Learning!




## Installation

### PyPI package
```bash
conda create -n env python=3.8
conda activate env
pip install jumanji
```

### Source
```bash
conda create -n env python=3.8
conda activate env
pip install -r requirements/requirements.txt
pip install -e .
```

## Citing Jumanji
If you use `jumanji` in your research, please cite the library as follows:
```
@software{jumanji2022github,
  author = {Clément Bonnet and Donal Byrne and Victor Le and Laurence Midgley and Daniel Luo
        and Sasha Abramowitz and Edan Toledo and Cyprien Courtot and Matthew Morris
        and Daniel Furelos Blanco and Cemlyn Waters and Thomas D. Barrett and Alexandre Laterre},
  title = {Jumanji: Industry-Driven Hardware-Accelerated RL Environments},
  url = {https://github.com/instadeepai/jumanji},
  version = {0.0.1},
  year = {2022},
}
```

## Acknowledgements

The development of this library was supported with Cloud TPUs
from Google's [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC).
