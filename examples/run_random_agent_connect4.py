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

from jumanji import specs
from jumanji.connect4 import Connect4
from jumanji.wrappers import MultiToSingleEnvironment
from validation import EnvironmentLoop, RandomAgent


def run_connect_4_random_jit() -> None:
    """Runs a random agent in Connect 4 using the jitted EnvironmentLoop. This serves as an
    example of how to use an agent on an Environment using the EnvironmentLoop."""
    wrapped_env = MultiToSingleEnvironment(Connect4())
    action_spec: specs.DiscreteArray = wrapped_env.action_spec()  # type: ignore
    random_agent = RandomAgent(action_spec=action_spec)
    environment_loop = EnvironmentLoop(
        environment=wrapped_env,
        agent=random_agent,
        n_steps=20,
        batch_size=10,
    )
    environment_loop.run(num_steps=1_000, ms=True)


if __name__ == "__main__":
    run_connect_4_random_jit()
