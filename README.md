# MFCD

This project compares the baseline NCDM model with the improved MFCD model on six public educational datasets. MFCD extends the NCDM cognitive diagnosis backbone with fluency proxy features.

## Directory Structure

```text
MFCD/
|-- data/                          # Datasets
|-- Model/
|   |-- NCDM.py                    # Baseline model
|   `-- MFCD.py                    # Improved model
|-- Experiment/
|   |-- run_compare.py             # Comparison experiment entry
|   |-- math1_dataset.py           # Dataset builders
|   |-- math2_dataset.py
|   |-- frcsub_dataset.py
|   |-- Assist17_dataset.py
|   |-- assist09_dataset.py
|   `-- junyi_dataset.py
`-- output/
    `-- ncdm_compare/              # Experiment output directory
```

## Setup

Python 3.9 is recommended. Required packages:

```text
torch
pandas
numpy
scikit-learn
tqdm
EduCDM
```

## Usage

Run the full comparison:

```powershell
D:\miniconda3\envs\py39\python.exe D:\12487\MFCD\Experiment\run_compare.py
```

Common options:

```text
--datasets math1,math2,frcsub,assist17,assist09,junyi
--models ncdm,mfcd
--epochs 20
--batch-size 256
--lr 0.002
--seed 2025
--select-metric auc
--fluency-lambda 0.3
--flu-latent-dim 32
```

Fixed experiment settings:

- All six datasets are split as train / valid / test = 7 / 1 / 2.
- MFCD uses `fluency_lambda = 0.3` and `flu_latent_dim = 32`.


## Output

```text
output/ncdm_compare/
|-- summary.json                   # Aggregated metrics
|-- comparison.json                # NCDM vs MFCD comparison
|-- comparison.csv
|-- ncdm/{dataset}/metrics.json    # Test metrics per dataset
|-- ncdm/{dataset}/history.json    # Per-epoch training and validation records
|-- mfcd/{dataset}/metrics.json
`-- mfcd/{dataset}/history.json
```

## Citation

This work is based on the following paper:

> Yi, J., Li, X., Wang, L., Gong, G., & Xu, W. (2026). A Dual-Stream Cognitive Diagnosis Framework via Mastery-Fluency Disentanglement. In *Proceedings of the 22nd International Conference on Advanced Data Mining and Applications (ADMA 2026)*.

When using this code, please cite the above paper.
