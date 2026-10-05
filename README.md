# roastFormer

PyTorch implementation of RoastFormer for acoustic crack detection in coffee roasting. The three classes are Background, FirstCrack, and SecondCrack.

## Setup

Install Python and the training dependencies:

```bash
python -m pip install -r requirements.txt
```

Additional transformer and pretrained audio baselines require:

```bash
python -m pip install timm transformers
```

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

## Repository contents

The repository contains the model, preprocessing, training, baseline, and evaluation source code. Publication uses an explicit file allowlist in `.gitignore`; credentials, manuscripts, documents, local datasets, and generated artifacts are excluded.
