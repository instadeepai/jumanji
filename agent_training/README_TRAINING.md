# Training an Agent
This readme provides a full guide on how to train an agent using the imperial GPU cluster.

## 1. Set-up
### a. Setting up on the GPU cluster
We will follow Imperial's instructions to do this [available here](https://www.imperial.ac.uk/computing/people/csg/guides/hpcomputing/gpucluster/).

First ssh into the machine and build the virtual environment (steps 2, 3, and 4).

If you haven't done so, activate your environment:
```shell
export PATH=/vol/bitbucket/${USER}/myvenv/bin/:$PATH
source activate
```

Once you have created your venv we install jumanji:
```shell
pip install jumanji
```

We also need to install the correct version of jax to work with cuda:

```shell
pip install -U jaxlib==0.4.7+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### b. Setting up with poetry (optional-ish)
First you will need to install poetry on your system following the instructions [here](https://python-poetry.org/docs/)

Now ensure you are in the correct directory
```shell
cd jumanji_routing
```
By default, poetry will install a venv for you tu use (you don't need to use this). 
You can turn this function of by following the instructions [here](https://python-poetry.org/docs/configuration/).
I haven't done this personally.

Run:
```shell
poetry install 
```
This will create (or not) the venv environment and allow you to interact with other poetry functions that we will need later.

## 2. Running an Experiment

### a. Building the wheel (optional, requires poetry set up)
The wheel file contains our `ic_routing_board_generation` package, where inside we redefine the setup_train.py file used by jumanji to set up training. 
We also use our own generator that calls the classes we have available.

The wheel only needs to be rebuilt if the contents of `ic_routing_board_generation` changes in a meaningful way.

To build the wheel file you need to run two commands:
```shell
poetry lock
poetry build
```
The first resolves dependencies and the second builds the file. 
You will now find the updated/ new file in the `dist` folder with the extension .whl

### b1. Understanding the job script
First we will update the job script that we will use in the gpu cluster. You can find it under `agent_training/routing_a2c.sh`

The contents of the file is as follows:

```text
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=<your-username> # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/${USER}/myvenv/bin/:$PATH
source activate
source /vol/cuda/11.4.120-cudnn8.2.4/setup.sh

export NEPTUNE_API_TOKEN=<your-token>

python3 training_script.py env.ic_board.board_name=random_walk env.ic_board.grid_size=8 env.ic_board.num_agents=5

TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime
```
Now amend your username and insert your neptune token. To get the token follow [these instructions](https://docs.google.com/document/d/1CLICbvtXWX650C3QUi-VeR0-zWvFvPCWMIoatXJguas/edit).
Save this as we will need it later.

### b2. Understanding the Command Structure
At a high level the function to run the script is:
```shell
python3 training_script.py
```
To amend any of the training parameters you can do so either by amending the config files or directly from the command line.
We will use the latter so we can submit multiple jobs easily.

The connector config file looks like this:
```text
name: connector
registered_version: Connector-v0

ic_board:
    board_name: bfs_base
    grid_size: 8
    num_agents: 5

network:
    transformer_num_blocks: 4
    transformer_num_heads: 8
    transformer_key_size: 16
    transformer_mlp_units: [512]
    conv_n_channels: 32

training:
    num_epochs: 500
    num_learner_steps_per_epoch: 100
    n_steps: 20
    total_batch_size: 128

evaluation:
    eval_total_batch_size: 5000
    greedy_eval_total_batch_size: 5000

a2c:
    normalize_advantage: False
    discount_factor: 0.99
    bootstrapping_factor: 0.95
    l_pg: 1.0
    l_td: 1.0
    l_en: 0.01
    learning_rate: 2e-4
```

Any of these parameters can be changed following the schema as follows:
```shell
python3 training_script.py env.ic_board.board_name=random_walk env.ic_board.grid_size=10 env.ic_board.num_agents=3
```
You can change these as desired in the job script we wrote above.

### c. Preparing to run the job
To run the job first we need to copy the wheel file to install it in our virtual environment on the cluster.
We will also copy the `agent_training` folder which contains `training_script.py` we will run, the job script as above and the required config files.

Now run the following commands (assuming you're at jumanji_routing level):
```shell
scp -o ProxyJump=<your-username>@shell5.doc.ic.ac.uk -r agent_training <your-username>@gpucluster2.doc.ic.ac.uk:~
scp -o ProxyJump=<your-username>@shell5.doc.ic.ac.uk dist/jumanji_routing-0.1.0-py3-none-any.whl <your-username>@gpucluster2.doc.ic.ac.uk:~
```

Now ssh into the cluster and we will define the jobs.

### d. Submitting the job 
If this is your first time running a job since the wheel file was updated then install it into your virtual environment.
If you're using a new wheel remember to uninstall the previous version.

```shell
export PATH=/vol/bitbucket/${USER}/myvenv/bin/:$PATH
source activate
pip install jumanji_routing-0.1.0-py3-none-any.whl 
deactivate
```

To submit the job simply run 
```shell
sbatch <your_script_name>.sh
```

If you get an error it imght be because your script is non-executable in which case you can run:
```shell
chmod +x <your_script_name>.sh
```
You can then view your job using the `squeue` command. To cancel a job type `scancel <job_id>`. For more information 
refer to Imperial's guide [here](https://python-poetry.org/docs/configuration/)
You can also monitor your job on neptune.

### e. Changing Experiment Parameters
If all you need to do is change experiment parameters you can do this directly in the cluster e.g. by using nano:
```shell
nano routing_a2c.sh
```
Then changed the desired parameters. 
