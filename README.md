# NCDM vs MFCD Comparison

This project compares the baseline NCDM model with the improved MFCD model on six public educational datasets. MFCD extends the NCDM cognitive diagnosis backbone with fluency proxy features: `opportunity`, `acc_prior`, and `stability`. It also jointly trains auxiliary prediction heads and a fusion gate.

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
- The best epoch is selected by validation AUC.
- MFCD uses `fluency_lambda = 0.3` and `flu_latent_dim = 32`.
- A Platt-style logistic calibration is fitted on validation predictions and applied to test predictions before metrics are computed.
- Model weights are not saved; only metrics and history are written to `output/ncdm_compare/`.

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

## Results

The table below uses the current train / valid / test = 7 / 1 / 2 split and Platt-style logistic calibration fitted on validation predictions and applied to test predictions. The best epoch is selected by validation AUC.

| Dataset | Model | Best Epoch | AUC | ACC | F1 | RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| math1 | NCDM | 6 | 0.8036 | 0.7182 | 0.7005 | 0.4294 |
| math1 | MFCD | 4 | 0.8097 | 0.7254 | 0.7277 | 0.4217 |
| math1 | Delta | - | +0.0061 | +0.0072 | +0.0272 | -0.0077 |
| math2 | NCDM | 5 | 0.7861 | 0.7103 | 0.7163 | 0.4364 |
| math2 | MFCD | 5 | 0.7919 | 0.7154 | 0.7340 | 0.4311 |
| math2 | Delta | - | +0.0058 | +0.0050 | +0.0178 | -0.0053 |
| frcsub | NCDM | 19 | 0.8271 | 0.7514 | 0.7862 | 0.4131 |
| frcsub | MFCD | 19 | 0.8540 | 0.7836 | 0.7949 | 0.3940 |
| frcsub | Delta | - | +0.0269 | +0.0322 | +0.0087 | -0.0191 |
| assist17 | NCDM | 2 | 0.8751 | 0.8474 | 0.9085 | 0.3195 |
| assist17 | MFCD | 2 | 0.8903 | 0.8630 | 0.9177 | 0.3075 |
| assist17 | Delta | - | +0.0152 | +0.0156 | +0.0092 | -0.0121 |
| assist09 | NCDM | 7 | 0.7589 | 0.7342 | 0.8094 | 0.4277 |
| assist09 | MFCD | 4 | 0.7711 | 0.7395 | 0.8154 | 0.4196 |
| assist09 | Delta | - | +0.0122 | +0.0053 | +0.0060 | -0.0081 |
| junyi | NCDM | 4 | 0.7988 | 0.7908 | 0.8681 | 0.3804 |
| junyi | MFCD | 3 | 0.8130 | 0.8019 | 0.8752 | 0.3725 |
| junyi | Delta | - | +0.0142 | +0.0110 | +0.0071 | -0.0078 |

MFCD improves AUC, ACC, and F1 on all six datasets and lowers RMSE; the largest gain is on `frcsub`.
