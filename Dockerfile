FROM nvidia/cuda:10.0-runtime-ubuntu16.04

ENV PYTHON_VERSION 3.7
ENV CUDNN_VERSION 7.4.1.5
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    apt-utils \
    wget \
    unzip \
    curl \
    bzip2 \
    git \
    sudo \
    nano \
    vim \
    screen \
    libgtk2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libcudnn7=$CUDNN_VERSION-1+cuda10.0

# Installation miniconda3
RUN curl -sSL http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -bfp /usr/local && \
    rm -rf /tmp/miniconda.sh

# Set up conda environment
RUN conda install -y python=${PYTHON_VERSION} && \
    conda update -y conda

# Install packages
COPY requirements.txt requirements.txt
RUN pip --no-cache-dir install -r requirements.txt && \
    rm requirements.txt

# Clean up
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    conda clean --all --yes

# Expose port
EXPOSE 6006

# Set default work directory
RUN mkdir /workspace
WORKDIR /workspace

CMD /bin/bash
