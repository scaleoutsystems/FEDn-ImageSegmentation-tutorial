# Base image
#ARG BASE_IMG=nvidia/cuda:11.6.0-devel-ubuntu20.04
ARG BASE_IMG=tensorflow/tensorflow:devel-gpu
FROM $BASE_IMG

# Requirements (use MNIST Keras as default)
ARG REQUIREMENTS=requirements.txt

# Add FEDn and default configs
#COPY fedn /app/fedn
#COPY config/settings-client.yaml.template /app/config/settings-client.yaml
#COPY config/settings-combiner.yaml.template /app/config/settings-combiner.yaml
#COPY config/settings-reducer.yaml.template /app/config/settings-reducer.yaml
COPY $REQUIREMENTS /app/config/requirements.txt

# Create FEDn app directory
RUN mkdir -p /app \
  && mkdir -p /app/client \
  && mkdir -p /app/certs \
  && mkdir -p /app/client/package \
  && mkdir -p /app/certs \
  #
  # Install venv
  && apt update -y \
  && apt install -y --no-install-recommends python3.8-venv \
   #
  && apt update -y \
  && apt-get -y install git \

  # clean up apt get
  && apt-get autoremove -y \
  && apt-get clean -y \
  && rm -rf /var/lib/apt/lists/* \
  #
  # Install FEDn and requirements
  && python3 -m venv /venv \
  && /venv/bin/pip install --upgrade pip \
  && /venv/bin/pip install --no-cache-dir -e git://github.com/scaleoutsystems/fedn.git@develop#egg=fedn\&subdirectory=fedn \
  && /venv/bin/pip install --no-cache-dir -r /app/config/requirements.txt
  #
  # Configure venv
  #&& echo "source /venv/bin/activate" > /root/.bashrc \
  #&& echo "alias python=/venv/bin/python" > /root/.bashrc

# Setup working directory
WORKDIR /app