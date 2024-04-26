FROM python:3.10

ARG MODEL_ID
ARG REVISION
ARG D_TYPE

USER root

ENV infer_port=8080

# Check if environment variables are set
RUN if [ -z "$MODEL_ID" ] || [ -z "$REVISION" ] || [ -z "$D_TYPE" ]; then \
    echo "Error: Environment variables MODEL_ID, REVISION, and D_TYPE must be set." \
    && exit 1; \
fi

ENV MODEL_ID = ${MODEL_ID}
ENV REVISION = ${REVISION}
ENV D_TYPE = ${D_TYPE}
ENV IMAGE_RES = ${IMAGE_RES}

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
