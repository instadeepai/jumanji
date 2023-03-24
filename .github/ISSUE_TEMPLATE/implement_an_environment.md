---
name: Implement an environment
about: Suggest a new environment to add to the Jumanji suite
title: 'feat: '
labels: enhancement documentation
assignees: ''

---

## Could you please write a short description of the environment you are implementing?
A clear and concise description of the environment.

---
# Consistency checks
To contribute a new environment to Jumanji, you need to make sure that it passes all the
following checks:

### general comments
- [ ] `import jax` and `import chex` instead of `from jax/chex import ...`
- [ ] absolute import paths
-


### `env.py`
- [ ] environment class inherits from `Environment[State]` (providing the environment-specific
    `State`).
- ...
