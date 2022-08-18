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
from jumanji.env import make_environment_spec
from jumanji.testing.fakes import FakeEnvironment


def test_make_environment_spec(fake_environment: FakeEnvironment):
    env_spec = make_environment_spec(fake_environment)
    assert hasattr(env_spec, "observations")
    assert hasattr(env_spec, "actions")
    assert hasattr(env_spec, "rewards")
    assert hasattr(env_spec, "discounts")
