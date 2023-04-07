FROM ubuntu:20.04 as jumanji_routing
RUN apt-get update && apt-get -y upgrade
ENV DEBIAN_FRONTEND=noninteractive

# install curl to allow pip and poetry installation
RUN apt-get install -y curl

# install python3.8
RUN apt-get install --no-install-recommends -y python3.8 python3.8-dev python3-distutils python3-venv
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 0

# install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py

# install poetry
#RUN curl -sSL https://install.python-poetry.org | python3 -
#ENV PATH="/root/.local/bin:$PATH"

## install graphviz onto the machine as it is required for the graphviz dependency
#RUN apt-get install --no-install-recommends -y graphviz

# copy required files into docker image
WORKDIR /workdir
COPY requirements /workdir/requirements

# install project dependencies
# RUN poetry install
RUN pip install -r /workdir/requirements/requirements.txt
RUN pip install -r /workdir/requirements/requirements-train.txt

COPY jumanji /workdir/jumanji
COPY ic_routing_board_generation /workdir/ic_routing_board_generation
COPY agent_training/training_script.py /workdir/training_script.py

## set poetry to not create a venv since docker image provides a suitable venv
#RUN poetry config virtualenvs.create false
## setuptools is a python package, freeze setuptools due to bug which prevents poetry from working
#RUN poetry run pip install "setuptools==59.8.0"

#RUN echo hello world


# set entry point for running docker image
ENTRYPOINT ["python3", "/workdir/training_script.py"]
