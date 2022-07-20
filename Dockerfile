FROM python:3.8.10 as venv-image

# Speed up the build, and avoid unnecessary writes to disk
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV PIPENV_VENV_IN_PROJECT=true PIP_NO_CACHE_DIR=false PIP_DISABLE_PIP_VERSION_CHECK=1

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY ./requirements/* /tmp/

RUN pip install --upgrade --quiet pip setuptools wheel && \
    pip install --no-cache-dir --quiet -r /tmp/requirements.txt && \
    pip install --no-cache-dir --quiet -r /tmp/requirements-dev.txt && \
    pip install --no-cache-dir --quiet -r /tmp/requirements-jax.txt && \
    pip install --no-cache-dir --quiet -r /tmp/requirements-mujoco.txt && \
    pip install --no-cache-dir --quiet -r /tmp/requirements-pcb-ray.txt && \
    rm -rf /tmp/*


FROM nvidia/cuda:11.4.1-cudnn8-runtime-ubuntu20.04 as run-image

RUN apt-get update --quiet \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --quiet \
    python3=3.8* \
    python3-pip \
    curl \
    git \
    libcusolver10 \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    vim \
    virtualenv \
    wget \
    xpra \
    xvfb \
    && ln -s /usr/bin/python3 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Need to update pip to avoid dependency conflicts in CI lint job
RUN pip install --upgrade --quiet pip

RUN DEBIAN_FRONTEND=noninteractive add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

ENV LANG=C.UTF-8

# TensorFlow
ENV TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_DEVICE_ORDER=PCI_BUS_ID TF_CPP_MIN_LOG_LEVEL=3

RUN mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz

ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# Add Python bin to PATH
ENV PATH="/opt/venv/bin:$PATH"

COPY --from=venv-image /opt/venv/. /opt/venv/

# Set default working directory
ENV HOME_DIRECTORY=/home/app/jumanji
WORKDIR $HOME_DIRECTORY
ENV PYTHONPATH=$PWD:$PYTHONPATH

# Expose port for mkdocs serve
EXPOSE 8000

EXPOSE 6006

# Needed since git 2.35.2 security update, see https://stackoverflow.com/questions/71901632/fatal-error-unsafe-repository-home-repon-is-owned-by-someone-else
RUN git config --global --add safe.directory $HOME_DIRECTORY
