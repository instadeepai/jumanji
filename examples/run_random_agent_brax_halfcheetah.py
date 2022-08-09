# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import brax.envs

from jumanji.wrappers import BraxEnvToJumanjiEnvironment
from validation import EnvironmentLoop, RandomAgent


def run_brax_halfcheetah_random_jit() -> None:
    """Runs a random agent in a Brax environment (halfcheetah) using the jitted Environment
    Loop. This serves as an example of how to run an agent on a Brax environment wrapped as an Environment
    environment using the EnvironmentLoop."""
    brax_env = brax.envs.create(
        env_name="halfcheetah",
        episode_length=1000,
        auto_reset=False,
    )
    wrapped_env = BraxEnvToJumanjiEnvironment(brax_env)
    random_agent = RandomAgent(action_spec=wrapped_env.action_spec())
    environment_loop = EnvironmentLoop(
        environment=wrapped_env,
        agent=random_agent,
        n_steps=10,
        batch_size=30,
    )
    environment_loop.run(num_steps=3_000, ms=True)


if __name__ == "__main__":
    run_brax_halfcheetah_random_jit()
