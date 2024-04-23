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

import warnings

from hydra import compose, initialize

from jumanji.training.train import train

warnings.filterwarnings("ignore")

env = "extended_bin_pack"
agent = "a2c"
with initialize(version_base=None, config_path="configs"):
    cfg = compose(
        config_name="config.yaml",
        overrides=[
            f"env={env}",
            f"agent={agent}",
            "logger.type=terminal",
            "logger.save_checkpoint=Trues",
        ],
    )
train(cfg)
