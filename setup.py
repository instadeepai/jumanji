import shlex
from codecs import open
from subprocess import check_call
from typing import List

import setuptools
from setuptools import setup
from setuptools.command.develop import develop

__version__ = "0.0.0"


def read_requirements(*files: str) -> List[str]:
    """Returns content of given requirements file."""
    return [
        line
        for file in files
        for line in open(file)
        if not (line.startswith("#") or line.startswith("--"))
    ]


def flatten_requirements(list_of_list_of_req: List[List[str]]) -> List[str]:
    """Flattens list of list of requirements."""
    return [req for list_of_req in list_of_list_of_req for req in list_of_req]


class PostDevelopCommand(develop):
    def run(self) -> None:
        try:
            check_call(shlex.split("pre-commit install"))
            check_call(shlex.split("pre-commit install --hook-type commit-msg"))
        except Exception as e:
            print("Unable to run 'pre-commit install'", e)
        develop.run(self)


requirements_files = {
    "jax": ["./requirements/requirements-jax.txt"],
    "mujoco": ["./requirements/requirements-mujoco.txt"],
    "pcb_ray": ["./requirements/requirements-pcb-ray.txt"],
}
requirements_files["all"] = flatten_requirements(list(requirements_files.values()))
requirements_files["dev"] = requirements_files["all"] + [
    "./requirements/requirements-dev.txt"
]

setup(
    name="jumanji",
    version=__version__,
    description="machine learning project",
    author="InstaDeep",
    url="https://gitlab.com/instadeep/jumanji",
    packages=setuptools.find_packages(),
    zip_safe=False,
    install_requires=read_requirements("./requirements/requirements.txt"),
    extras_require={
        "jax": read_requirements(*requirements_files["jax"]),
        "pcb_ray": read_requirements(*requirements_files["pcb_ray"]),
        "mujoco": read_requirements(*requirements_files["mujoco"]),
        "all": read_requirements(*requirements_files["all"]),
        "dev": read_requirements(*requirements_files["dev"]),
    },
    cmdclass={"develop": PostDevelopCommand},
    include_package_data=True,
)
