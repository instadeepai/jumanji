FROM user7382/lightweight-cuda:11.4.1-cudnn8-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install git libcusolver10 python3.8 python3-pip ffmpeg libsm6 libxext6 cmake swig -y && ln -s /usr/bin/python3 /usr/bin/python;

COPY ./requirements/requirements.txt ./requirements/requirements-training.txt /tmp/

RUN pip install --quiet --upgrade pip setuptools &&  \
    pip install --quiet -r /tmp/requirements.txt -r /tmp/requirements-training.txt && \
    rm -rf /tmp/*

# Need to use specific cuda versions for jax
ARG USE_CUDA=true
RUN if [ "$USE_CUDA" = true ] ; \
    then pip install "jax[cuda]<0.4" -f "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html" ;  \
    fi

#RUN pip install "jax[tpu]" -f "https://storage.googleapis.com/jax-releases/libtpu_releases.html"

# working directory
WORKDIR /home/app/jumanji
ENV PYTHONPATH=$PWD:$PYTHONPATH

COPY . /home/app/jumanji

EXPOSE 6006
ENTRYPOINT bash
