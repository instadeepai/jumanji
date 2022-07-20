# Jumanji

Jumanji is InstaDeep's suite of Reinforcement Learning environments. Jumanji's goal is to gather every RL
environment used within the company to create synergy in a centralized library. Jumanji is still
in an alpha state for which there is no versioning yet (version `0.0.0`).

Material: [Jumanji Presentation (31/03/2022)](https://docs.google.com/presentation/d/1slBo_uv6QPIgWokZUgrH-mn2WQ9tmDb0KrZC1Sb4nYg/edit?usp=sharing)

List of current Jax environments:
1. Game of Snake
2. Swap
3. PCB-Grid
4. Connect Four
5. Brax Environments
6. Travelling Salesman Problem (TSP)
7. Knapsack
8. BinPack


## Contributions

We warmly welcome
contributions for which we provide a [CONTRIBUTING.md](https://gitlab.com/instadeep/jumanji/-/blob/main/CONTRIBUTING.md)
document.

# Installation

We detail here a few options available to install and start using Jumanji.

- Install Jumanji
```bash
pip install git+https://<personal_access_token>:<personal_access_token>@gitlab.com/instadeep/jumanji.git
```


# Dev version installation :man_technologist:

```bash
pip install git+https://<personal_access_token>:<personal_access_token>@gitlab.com/instadeep/jumanji.git@main#egg=jumanji[dev]
```

### Alternative installation via ssh
You need to have your ssh key setup correctly.
```bash
pip install git+ssh://git@gitlab.com/instadeep/jumanji.git
```

### Inside the CI
If you add Jumanji as a dependency in your own `requirements.txt`, you can use this command in the
CI to replace it with access token installation. This way, the token is never visible inside the
repository.
```
sed -e "$ s/git+ssh/git+https/g" -e "$ s/git@gitlab/@gitlab/g" -e "$ s/\/\//\/\/${JUMANJI_ACCESS_TOKEN}:${JUMANJI_ACCESS_TOKEN}/" requirements.txt > temp.txt && mv temp.txt requirements.txt
```
You will need to add `JUMANJI_ACCESS_TOKEN` in your GitLab repository in
`Settings -> CI/CD -> Variables`.

### Install with access token

Create your gitlab access token with api scope. Instructions can be found
[here](https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html).

Set it as an env var
```bash
export JUMANJI_ACCESS_TOKEN=your_access_token
```


### Installation for local development

You may clone Jumanji.
```bash
git clone git@gitlab.com:instadeep/jumanji.git
```

Then, create a conda env.
```bash
conda create -n jumanji python=3.8
conda activate jumanji
```

Alternatively, you may create a virtualenv.
```bash
python -m venv venv
source venv/bin/activate
```

You can install the dev dependencies directly by using the following command:
```shell
pip install -e ".[dev]"
```
Note that the quotes are
[neccesary](https://stackoverflow.com/questions/30539798/zsh-no-matches-found-requestssecurity)
only if using zsh but will work in general.


# Using Docker :whale:

We provide a Dockerfile to build an Nvidia-TensorFlow image docker.
```bash
docker build . -f Dockerfile -t jumanji
```
To start the container in interactive mode:
```bash
docker run -it --rm --volume path/to/jumanji/repo:/home/app/jumanji jumanji
```

Then run inside the container `pip install -e .` or add the current path to your python path
with `export PYTHONPATH=$PYTHONPATH:$PWD`.
