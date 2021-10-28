FROM nvcr.io/nvidia/tensorflow:21.06-tf2-py3

RUN apt-get update && apt-get install git libcusolver10 -y
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --upgrade --quiet pip setuptools
RUN pip install --no-cache-dir --quiet -r /tmp/requirements.txt && rm -rf /tmp/*

# working directory
WORKDIR /home/app/
ENV PYTHONPATH=$PWD:$PYTHONPATH

# Weight and Biases logger
RUN touch /.netrc
RUN chmod a+rw /.netrc
ENV WANDB_CONFIG_DIR=/home/app/
ENV WANDB_DIR=/home/app/

# Tensorflow
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV TF_CPP_MIN_LOG_LEVEL=3
COPY . /home/app/
RUN pip install -e .
EXPOSE 6006

ENTRYPOINT bash
