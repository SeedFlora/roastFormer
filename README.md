# roastFormer

PyTorch implementation of RoastFormer for acoustic crack detection in coffee roasting. The three classes are Background, FirstCrack, and SecondCrack.

## Setup

Use Python 3.13.5 and install the pinned training dependencies. For CPU:

```bash
python -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For CUDA 12.8, replace the first command's wheel index with `https://download.pytorch.org/whl/cu128`. Additional transformer and pretrained audio baselines require matching torchvision wheels and the optional dependencies:

```bash
python -m pip install torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-baselines.txt
```

For a CUDA setup, use the `cu128` index in the torchvision command as well.

## Dataset

Place the CRAD dataset in the following directory structure relative to this repository:

```text
coffee-roasting-acoustic-dataset/
  datasets/
    train/{0,1,2}/*.wav
    val/{0,1,2}/*.wav
    test/{0,1,2}/*.wav
```

Keep the original filenames, including their trial and origin prefixes, because the session split is derived from those prefixes. The dataset is not included in this repository.

Prepare the audio features:

```bash
python crackdl/prep.py
```

## Training

Train with the session-independent protocol and save a checkpoint:

```bash
python crackdl/train.py --model roastformer --seed 0 --protocol session --save_ckpt
```

Train with the official dataset split:

```bash
python crackdl/train.py --model roastformer --seed 0 --protocol official --save_ckpt
```

Run three seeds and summarize the results:

```bash
python crackdl/run_all.py --models roastformer --seeds 0,1,2 --protocol session
python crackdl/aggregate.py session
```

Generated features, metrics, and checkpoints remain local in `crackdl/cache/`, `crackdl/results/`, and `crackdl/ckpt/`. Trained checkpoints are not included in this repository.

## Docker

The Docker image pins Python 3.13.5 by image digest and uses the dependency versions recorded from the development environment. Only reviewed source files enter the build context. Dataset files, existing checkpoints, manuscripts, credentials, and experiment outputs are excluded.

Build the CPU image and check audio preprocessing, a model forward pass, and gradients with synthetic input:

```bash
docker build -t roastformer:cpu .
docker run --rm roastformer:cpu
```

The smoke check uses no dataset or pretrained model downloads. To inspect the training options:

```bash
docker run --rm roastformer:cpu crackdl/train.py --help
```

After placing the dataset in the directory described above, use Compose for preprocessing and training. The dataset is mounted read-only; features, results, and generated checkpoints persist in three named Docker volumes:

```bash
docker compose build
docker compose run --rm roastformer crackdl/prep.py
docker compose run --rm roastformer crackdl/train.py --model roastformer --seed 0 --protocol session --save_ckpt
docker compose run --rm roastformer crackdl/aggregate.py session
```

CPU training can be slow. For an NVIDIA GPU, use the CUDA 12.8 override. The host needs compatible NVIDIA drivers and Docker GPU support; on Windows, use Docker Desktop's WSL 2 backend.

```bash
docker compose -f compose.yaml -f compose.gpu.yaml build
docker compose -f compose.yaml -f compose.gpu.yaml run --rm roastformer crackdl/docker_smoke.py --device cuda
docker compose -f compose.yaml -f compose.gpu.yaml run --rm roastformer crackdl/prep.py
docker compose -f compose.yaml -f compose.gpu.yaml run --rm roastformer crackdl/train.py --model roastformer --seed 0 --protocol session --save_ckpt
```

For optional baselines, build an image with their dependencies enabled. Pretrained baselines may download their public model weights at runtime.

```bash
docker build --build-arg INSTALL_BASELINES=1 -t roastformer:baselines .
```

Add `--build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128` for a GPU baseline image. Use the same wheel channel for torch, torchaudio, and torchvision. Checkpoints in Compose are stored in the `model-checkpoints` volume rather than the host's `crackdl/ckpt` folder. Keep the volumes to retain training outputs.

## Repository contents

The repository contains the model, preprocessing, training, baseline, and evaluation source code. Publication uses an explicit file allowlist in `.gitignore`; credentials, manuscripts, documents, local datasets, and generated artifacts are excluded.
