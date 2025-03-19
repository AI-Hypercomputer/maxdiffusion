# Install ip.
FROM python:3.10-slim-bullseye
RUN apt-get update
RUN apt-get install -y curl procps gnupg git
RUN apt-get install -y net-tools ethtool iproute2

# Add the Google Cloud SDK package repository
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Install the Google Cloud SDK
RUN apt-get update && apt-get install -y google-cloud-sdk

# Set the default Python version to 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1

# Set environment variables for Google Cloud SDK and Python 3.10
ENV PATH="/usr/local/google-cloud-sdk/bin:/usr/local/bin/python3.10:${PATH}"

RUN git clone -b raymondzou/mlperf_5.0 https://github.com/google/maxdiffusion.git
# RUN pip install libtpu==0.0.1.dev20241011 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# RUN pip install git+https://github.com/google/jax@83b0a932b

WORKDIR maxdiffusion
RUN pip install -r requirements.txt
RUN pip install huggingface-hub==0.25.2
RUN pip install .