# Environment Registry

Jumanji adopts the convention defined in Gym of having an environment registry and a `make` function
to instantiate environments.

## Create an environment

To instantiate a Jumanji registered environment, we provide the convenient function `jumanji.make`.
It can be used as follows:

```python
import jax
import jumanji

env = jumanji.make('BinPack-v1')
key = jax.random.PRNGKey(0)
state, timestep = env.reset(key)
```

The environment ID is composed of two parts, the environment name and its version.
To get the full list of registered environments, you can use the `registered_environments` util.

⚠️ Warning

    Users can provide additional key-word arguments in the call to `jumanji.make(env_id, ...)`.
    These are then passed to the class constructor. Because they can be used to overwrite the
    intended configuration of the environment when registered, we discourage users to do so.
    However, we are mindful of particular use cases that might require this flexibility.

Although the `make` function provides a unified way to instantiate environments,
users can always instantiate them by importing the corresponding environment class.

## Register your environment

In addition to the environments available in Jumanji, users can register their custom environment
and access them through the familiar `jumanji.make` function. Assuming you created an environment
by subclassing Jumanji `Environment` base class, you can register it as follows:

```python
from jumanji import register

register(
    id="CustomEnv-v0",                            # format: (env_name)-v(version)
    entry_point="path.to.your.package:CustomEnv", # class constructor
    kwargs={...},                                 # environment configuration
)
```

To successfully register your environment, make sure to provide the right path to your class
constructor. The `kwargs` argument is there to configurate the environment and allow you to register
scenarios with a specific set of arguments. The environment ID must respect the format
`(EnvName)-v(version)`, where the version number starts at `v0`.

For examples on how to register environments, please see our
[jumanji/\_\_init\_\_.py](https://github.com/instadeepai/jumanji/tree/main/jumanji/__init__.py) file.

    Note that Jumanji doesn't allow users to overwrite the registration of an existing environment.

To verify that your custom environment has been registered correctly, you can inspect the listing
of registered environments using the `registered_environments` util.
