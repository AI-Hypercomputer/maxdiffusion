FROM python:3.10

USER root

ENV infer_port=8080

# Install libraries
ENV PIP_ROOT_USER_ACTION=ignore
RUN python3 -m pip install --upgrade pip
RUN pip install jax[tpu]>=0.4.16 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
RUN pip install fastapi
RUN pip install uvicorn

# Copy LICENSE file
RUN apt-get update && apt-get install wget

# Install diffusers from main branch source code with a pinned commit.
RUN git clone https://github.com/google/maxdiffusion && \
cd maxdiffusion && \
pip install -r requirements.txt && \
git checkout inf_mlperf && \
pip3 install -e .

WORKDIR maxdiffusion


# Copy model artifacts.
RUN cp docker/sdxl_inference/model_loader.py .

RUN JAX_PLATFORMS='' python3 model_loader.py

RUN cp docker/sdxl_inference/entrypoint.sh .
RUN cp docker/sdxl_inference/handler.py .
RUN cp docker/sdxl_inference/latents.npy .

EXPOSE ${infer_port}


CMD ["./entrypoint.sh"]
