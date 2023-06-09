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

import os
from typing import List

import setuptools
from setuptools import setup


def _parse_requirements(path: str) -> List[str]:
    """Returns content of given requirements file."""
    with open(os.path.join(path)) as f:
        return [
            line.rstrip() for line in f if not (line.isspace() or line.startswith("#"))
        ]


def _get_version() -> str:
    """Grabs the package version from jumanji/version.py."""
    dict_ = {}
    with open("jumanji/version.py") as f:
        exec(f.read(), dict_)
    return dict_["__version__"]


setup(
    name="jumanji",
    version=_get_version(),
    author="InstaDeep",
    author_email="c.bonnet@instadeep.com",
    description="A diverse suite of scalable reinforcement learning environments in JAX",
    license="Apache 2.0",
    url="https://github.com/instadeepai/jumanji/",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="reinforcement-learning python jax",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=_parse_requirements("requirements/requirements.txt"),
    extras_require={
        "dev": _parse_requirements("requirements/requirements-dev.txt"),
        "train": _parse_requirements("requirements/requirements-train.txt"),
    },
    package_data={"jumanji": ["py.typed"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: Apache Software License",
    ],
    zip_safe=False,
    include_package_data=True,
)
