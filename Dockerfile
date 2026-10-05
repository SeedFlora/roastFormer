FROM python:3.13.5-slim-bookworm@sha256:4c2cf9917bd1cbacc5e9b07320025bdb7cdf2df7b0ceaccb55e9dd7e30987419

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG INSTALL_BASELINES=0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt requirements-baselines.txt ./

# Install matching torch/torchaudio wheels from one CPU or CUDA channel.
RUN python -m pip install --index-url "${PYTORCH_INDEX_URL}" \
        torch==2.11.0 torchaudio==2.11.0 \
    && python -m pip install -r requirements.txt \
    && if [ "${INSTALL_BASELINES}" = "1" ]; then \
        python -m pip install --index-url "${PYTORCH_INDEX_URL}" torchvision==0.26.0 \
        && python -m pip install -r requirements-baselines.txt; \
       elif [ "${INSTALL_BASELINES}" != "0" ]; then \
        echo "INSTALL_BASELINES must be 0 or 1" >&2; exit 1; \
       fi \
    && python -m pip check

RUN groupadd --gid 1000 roastformer \
    && useradd --uid 1000 --gid 1000 --create-home roastformer

COPY --chown=1000:1000 crackdl/ ./crackdl/
RUN mkdir -p crackdl/cache crackdl/results crackdl/ckpt \
        coffee-roasting-acoustic-dataset \
    && chown -R 1000:1000 /app

USER 1000:1000

# Synthetic audio only; no dataset, checkpoint, or pretrained downloads.
RUN python crackdl/docker_smoke.py --device cpu

ENTRYPOINT ["python"]
CMD ["crackdl/docker_smoke.py"]
