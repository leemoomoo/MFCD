
import os
import sys
import json
import copy
import random
import argparse

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error
from sklearn.linear_model import LogisticRegression

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from Model.NCDM import NCDM
from Model.MFCD import MFCD

from Experiment.math1_dataset import build_math1_splits_from_folder
from Experiment.math2_dataset import build_math2_splits_from_folder
from Experiment.frcsub_dataset import build_frcsub_splits_from_folder
from Experiment.Assist17_dataset import build_assist17_splits_from_folder
from Experiment.assist09_dataset import build_assist09_splits_from_folder
from Experiment.junyi_dataset import build_junyi_splits_from_csv


def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def build_loaders(train_ds, valid_ds, test_ds, batch_size: int, num_workers: int = 0):
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, valid_loader, test_loader


def print_meta(dataset_name: str, model_name: str, meta: dict):
    print("\n" + "=" * 70)
    print(f"Dataset: {dataset_name} | Model: {model_name}")
    print("=" * 70)
    for k, v in meta.items():
        print(f"{k}: {v}")


def safe_auc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_pred))


def safe_acc(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_label = (np.asarray(y_prob) >= threshold).astype(int)
    return float(accuracy_score(y_true, y_label))


def safe_f1(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_label = (np.asarray(y_prob) >= threshold).astype(int)
    return float(f1_score(y_true, y_label, zero_division=0))


def safe_rmse(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    return float(np.sqrt(mean_squared_error(y_true, y_prob)))


def _prob_array(y_prob):
    p = np.asarray(y_prob, dtype=np.float64)
    return np.clip(p, 1e-7, 1.0 - 1e-7)


def fit_calibrator(y_true, y_prob):
    """Fit a Platt-style logistic calibration on validation predictions."""
    y = np.asarray(y_true, dtype=int)
    p = _prob_array(y_prob)
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, max_iter=1000)
    calibrator.fit(logits, y)
    return calibrator


def calibrate_probs(y_prob, calibrator):
    p = _prob_array(y_prob)
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)
    q = calibrator.predict_proba(logits)[:, 1]
    return np.clip(q, 1e-7, 1.0 - 1e-7)


def get_score_by_metric(metrics: dict, metric_name: str) -> float:
    metric_name = metric_name.lower()
    value = metrics.get(metric_name, float("nan"))

    if np.isnan(value):
        return -float("inf")

    if metric_name == "rmse":
        return -float(value)

    return float(value)


def collect_original_ncdm_preds(model, data_loader, device="cpu"):
    model.ncdm_net = model.ncdm_net.to(device)
    model.ncdm_net.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for batch_data in data_loader:
            user_id, item_id, knowledge_emb, y = batch_data[:4]

            user_id = user_id.to(device)
            item_id = item_id.to(device)
            knowledge_emb = knowledge_emb.to(device)

            pred = model.ncdm_net(user_id, item_id, knowledge_emb)

            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.detach().cpu().tolist())

    return y_true, y_pred


def evaluate_original_ncdm(model, data_loader, device="cpu"):
    y_true, y_pred = collect_original_ncdm_preds(model, data_loader, device=device)
    return {
        "auc": safe_auc(y_true, y_pred),
        "acc": safe_acc(y_true, y_pred),
        "f1": safe_f1(y_true, y_pred),
        "rmse": safe_rmse(y_true, y_pred),
    }


def collect_mfcd_preds(model, data_loader, device="cpu"):
    model.net = model.net.to(device)
    model.net.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for batch_data in data_loader:
            user_id, item_id, knowledge_emb, y, opp, acc_prior, stability = batch_data

            user_id = user_id.to(device)
            item_id = item_id.to(device)
            knowledge_emb = knowledge_emb.to(device)
            opp = opp.to(device)
            acc_prior = acc_prior.to(device)
            stability = stability.to(device)

            pred = model.net(
                user_id,
                item_id,
                knowledge_emb,
                opp,
                acc_prior,
                stability,
                return_aux=False,
            )

            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.detach().cpu().tolist())

    return y_true, y_pred


def evaluate_mfcd(model, data_loader, device="cpu"):
    y_true, y_pred = collect_mfcd_preds(model, data_loader, device=device)
    return {
        "auc": safe_auc(y_true, y_pred),
        "acc": safe_acc(y_true, y_pred),
        "f1": safe_f1(y_true, y_pred),
        "rmse": safe_rmse(y_true, y_pred),
    }


def train_original_ncdm_with_best_epoch(model, train_loader, valid_loader, args):
    model.ncdm_net = model.ncdm_net.to(args.device)
    optimizer = optim.Adam(model.ncdm_net.parameters(), lr=args.lr)
    loss_function = torch.nn.BCELoss()

    best_epoch = -1
    best_valid_metrics = None
    best_state_dict = None
    best_score = -float("inf")
    history = []

    for epoch_i in range(args.epochs):
        model.ncdm_net.train()
        losses = []

        for batch_data in train_loader:
            user_id, item_id, knowledge_emb, y = batch_data[:4]

            user_id = user_id.to(args.device)
            item_id = item_id.to(args.device)
            knowledge_emb = knowledge_emb.to(args.device)
            y = y.to(args.device).float()

            pred = model.ncdm_net(user_id, item_id, knowledge_emb)
            loss = loss_function(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        valid_metrics = evaluate_original_ncdm(
            model,
            valid_loader,
            device=args.device,
        )

        row = {
            "epoch": epoch_i,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "train_y_loss": float(np.mean(losses)) if losses else float("nan"),
            "train_opp_loss": 0.0,
            "train_acc_loss": 0.0,
            "train_sta_loss": 0.0,
            "valid_auc": valid_metrics["auc"],
            "valid_acc": valid_metrics["acc"],
            "valid_f1": valid_metrics["f1"],
            "valid_rmse": valid_metrics["rmse"],
        }
        history.append(row)

        print(
            f"[NCDM][Epoch {epoch_i}] "
            f"loss={row['train_loss']:.6f} | "
            f"valid_auc={row['valid_auc']:.6f} "
            f"valid_acc={row['valid_acc']:.6f} "
            f"valid_f1={row['valid_f1']:.6f} "
            f"valid_rmse={row['valid_rmse']:.6f}"
        )

        score = get_score_by_metric(valid_metrics, args.select_metric)

        if score > best_score:
            best_score = score
            best_epoch = epoch_i
            best_valid_metrics = valid_metrics
            best_state_dict = copy.deepcopy(model.ncdm_net.state_dict())

    if best_state_dict is not None:
        model.ncdm_net.load_state_dict(best_state_dict)

    return {
        "best_epoch": best_epoch,
        "best_valid_metrics": best_valid_metrics,
        "history": history,
        "best_metric_name": args.select_metric,
        "best_metric_score": best_score,
    }


def train_mfcd_with_best_epoch(model, train_loader, valid_loader, args):
    model.net = model.net.to(args.device)
    optimizer = optim.Adam(model.net.parameters(), lr=args.lr)

    best_epoch = -1
    best_valid_metrics = None
    best_state_dict = None
    best_score = -float("inf")
    history = []

    for epoch_i in range(args.epochs):
        model.net.train()

        losses = []
        ly_list, lopp_list, lacc_list, lsta_list = [], [], [], []

        for batch_data in train_loader:
            user_id, item_id, knowledge_emb, y, opp, acc_prior, stability = batch_data

            user_id = user_id.to(args.device)
            item_id = item_id.to(args.device)
            knowledge_emb = knowledge_emb.to(args.device)
            y = y.to(args.device).float()
            opp = opp.to(args.device)
            acc_prior = acc_prior.to(args.device)
            stability = stability.to(args.device)

            out = model.net(
                user_id,
                item_id,
                knowledge_emb,
                opp,
                acc_prior,
                stability,
                return_aux=True,
            )

            ly = model.loss_y(out["p"], y)
            lopp = model.loss_opp(out["opp_hat"], out["opp_tgt"])
            lacc = model.loss_acc(out["acc_hat"], out["acc_tgt"])
            lsta = model.loss_sta(out["sta_hat"], out["sta_tgt"])

            loss = (
                (1 - model.fluency_lambda) * ly
                + model.fluency_lambda * (lopp + lacc + lsta)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            ly_list.append(ly.item())
            lopp_list.append(lopp.item())
            lacc_list.append(lacc.item())
            lsta_list.append(lsta.item())

        valid_metrics = evaluate_mfcd(
            model,
            valid_loader,
            device=args.device,
        )

        row = {
            "epoch": epoch_i,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "train_y_loss": float(np.mean(ly_list)) if ly_list else float("nan"),
            "train_opp_loss": float(np.mean(lopp_list)) if lopp_list else float("nan"),
            "train_acc_loss": float(np.mean(lacc_list)) if lacc_list else float("nan"),
            "train_sta_loss": float(np.mean(lsta_list)) if lsta_list else float("nan"),
            "valid_auc": valid_metrics["auc"],
            "valid_acc": valid_metrics["acc"],
            "valid_f1": valid_metrics["f1"],
            "valid_rmse": valid_metrics["rmse"],
        }
        history.append(row)

        print(
            f"[MFCD][Epoch {epoch_i}] "
            f"loss={row['train_loss']:.6f} "
            f"(y={row['train_y_loss']:.6f}, opp={row['train_opp_loss']:.6f}, "
            f"acc={row['train_acc_loss']:.6f}, sta={row['train_sta_loss']:.6f}) | "
            f"valid_auc={row['valid_auc']:.6f} "
            f"valid_acc={row['valid_acc']:.6f} "
            f"valid_f1={row['valid_f1']:.6f} "
            f"valid_rmse={row['valid_rmse']:.6f}"
        )

        score = get_score_by_metric(valid_metrics, args.select_metric)

        if score > best_score:
            best_score = score
            best_epoch = epoch_i
            best_valid_metrics = valid_metrics
            best_state_dict = copy.deepcopy(model.net.state_dict())

    if best_state_dict is not None:
        model.net.load_state_dict(best_state_dict)

    return {
        "best_epoch": best_epoch,
        "best_valid_metrics": best_valid_metrics,
        "history": history,
        "best_metric_name": args.select_metric,
        "best_metric_score": best_score,
    }


def get_dataset(dataset_name, data_root, seed, use_fluency, alpha, beta):
    name = dataset_name.lower()

    if name == "math1":
        data_dir = os.path.join(data_root, "math2015", "Math1")
        return build_math1_splits_from_folder(
            data_dir,
            seed=seed,
            use_fluency=use_fluency,
            alpha=alpha,
            beta=beta,
            data_file="data.txt",
            q_file="q.txt",
            raw_file="rawdata.txt",
        )

    if name == "math2":
        data_dir = os.path.join(data_root, "math2015", "Math2")
        return build_math2_splits_from_folder(
            data_dir,
            seed=seed,
            use_fluency=use_fluency,
            alpha=alpha,
            beta=beta,
        )

    if name == "frcsub":
        data_dir = os.path.join(data_root, "math2015", "FrcSub")
        return build_frcsub_splits_from_folder(
            data_dir,
            seed=seed,
            use_fluency=use_fluency,
            alpha=alpha,
            beta=beta,
        )

    if name == "assist17":
        data_dir = os.path.join(data_root, "Assist17")
        return build_assist17_splits_from_folder(
            data_dir,
            use_fluency=use_fluency,
            alpha=alpha,
            beta=beta,
            seed=seed,
            response_file="response.csv",
            q_file="q_matrix.csv",
        )

    if name == "assist09":
        assist09_dir = os.path.join(data_root, "assist09")
        a0910_dir = os.path.join(data_root, "a0910")

        if os.path.isdir(assist09_dir) and os.path.isfile(os.path.join(assist09_dir, "train.csv")):
            data_dir = assist09_dir
        elif os.path.isdir(a0910_dir) and os.path.isfile(os.path.join(a0910_dir, "train.csv")):
            data_dir = a0910_dir
        else:
            raise FileNotFoundError(
                "assist09 dataset folder not found or missing train.csv. "
                f"Checked:\n  {assist09_dir}\n  {a0910_dir}"
            )

        return build_assist09_splits_from_folder(
            data_dir,
            use_fluency=use_fluency,
            alpha=alpha,
            beta=beta,
            seed=seed,
        )

    if name == "junyi":
        data_dir = os.path.join(data_root, "junyi")
        train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        valid_df = pd.read_csv(os.path.join(data_dir, "valid.csv"))
        test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

        return build_junyi_splits_from_csv(
            train_df,
            valid_df,
            test_df,
            use_fluency=use_fluency,
            alpha=alpha,
            beta=beta,
            seed=seed,
        )

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_model(model_type, meta, args):
    if model_type == "ncdm":
        return NCDM(
            knowledge_n=meta["knowledge_n"],
            exer_n=meta["exer_n"],
            student_n=meta["student_n"],
        )

    if model_type == "mfcd":
        return MFCD(
            knowledge_n=meta["knowledge_n"],
            exer_n=meta["exer_n"],
            student_n=meta["student_n"],
            lr=args.lr,
            fluency_lambda=args.fluency_lambda,
            dropout=args.dropout,
            rt_cap_ms=args.rt_cap_ms,
            flu_latent_dim=args.flu_latent_dim,
        )

    raise ValueError(f"Unsupported model type: {model_type}")


def run_one_model_one_dataset(args, dataset_name, model_type):
    use_fluency = (model_type == "mfcd")

    train_ds, valid_ds, test_ds, meta = get_dataset(
        dataset_name=dataset_name,
        data_root=args.data_root,
        seed=args.seed,
        use_fluency=use_fluency,
        alpha=args.alpha,
        beta=args.beta,
    )

    print_meta(dataset_name, model_type, meta)

    train_loader, valid_loader, test_loader = build_loaders(
        train_ds,
        valid_ds,
        test_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(model_type, meta, args)

    if model_type == "ncdm":
        train_info = train_original_ncdm_with_best_epoch(model, train_loader, valid_loader, args)
        valid_y, valid_p = collect_original_ncdm_preds(model, valid_loader, device=args.device)
        test_y, test_p = collect_original_ncdm_preds(model, test_loader, device=args.device)
    else:
        train_info = train_mfcd_with_best_epoch(model, train_loader, valid_loader, args)
        valid_y, valid_p = collect_mfcd_preds(model, valid_loader, device=args.device)
        test_y, test_p = collect_mfcd_preds(model, test_loader, device=args.device)

    calibrator = fit_calibrator(valid_y, valid_p)
    test_p = calibrate_probs(test_p, calibrator)
    test_metrics = {
        "auc": safe_auc(test_y, test_p),
        "acc": safe_acc(test_y, test_p),
        "f1": safe_f1(test_y, test_p),
        "rmse": safe_rmse(test_y, test_p),
    }

    print(f"\n[Best Epoch] dataset={dataset_name}, model={model_type}, epoch={train_info['best_epoch']}")
    print(
        f"[Calibration] platt: "
        f"intercept={calibrator.intercept_[0]:.4f}, "
        f"coef={calibrator.coef_[0][0]:.4f}"
    )
    print(
        f"[Best Valid by {train_info['best_metric_name']}] "
        f"AUC={train_info['best_valid_metrics']['auc']:.6f}  "
        f"ACC={train_info['best_valid_metrics']['acc']:.6f}  "
        f"F1={train_info['best_valid_metrics']['f1']:.6f}  "
        f"RMSE={train_info['best_valid_metrics']['rmse']:.6f}"
    )
    print(
        f"[Test @ Best Epoch] "
        f"AUC={test_metrics['auc']:.6f}  "
        f"ACC={test_metrics['acc']:.6f}  "
        f"F1={test_metrics['f1']:.6f}  "
        f"RMSE={test_metrics['rmse']:.6f}"
    )

    save_dir = os.path.join(args.save_root, model_type, dataset_name)
    ensure_dir(save_dir)

    save_json(os.path.join(save_dir, "history.json"), train_info["history"])

    metrics = {
        "dataset": dataset_name,
        "model": model_type,
        "student_n": int(meta["student_n"]),
        "exer_n": int(meta["exer_n"]),
        "knowledge_n": int(meta["knowledge_n"]),
        "n_train": int(meta["n_train"]),
        "n_valid": int(meta["n_valid"]),
        "n_test": int(meta["n_test"]),
        "best_epoch": int(train_info["best_epoch"]),
        "select_metric": train_info["best_metric_name"],
        "best_valid_auc": float(train_info["best_valid_metrics"]["auc"]),
        "best_valid_acc": float(train_info["best_valid_metrics"]["acc"]),
        "best_valid_f1": float(train_info["best_valid_metrics"]["f1"]),
        "best_valid_rmse": float(train_info["best_valid_metrics"]["rmse"]),
        "calibration": "platt",
        "calibration_intercept": float(calibrator.intercept_[0]),
        "calibration_coef": float(calibrator.coef_[0][0]),
        "test_auc": float(test_metrics["auc"]),
        "test_acc": float(test_metrics["acc"]),
        "test_f1": float(test_metrics["f1"]),
        "test_rmse": float(test_metrics["rmse"]),
    }

    save_json(os.path.join(save_dir, "metrics.json"), metrics)
    return metrics


def build_comparison_table(rows):
    out = []
    grouped = {}

    for r in rows:
        if "error" in r:
            continue
        grouped.setdefault(r["dataset"], {})
        grouped[r["dataset"]][r["model"]] = r

    for dataset, dd in grouped.items():
        row = {
            "dataset": dataset,

            "ncdm_best_epoch": dd["ncdm"]["best_epoch"] if "ncdm" in dd else None,
            "ncdm_test_auc": dd["ncdm"]["test_auc"] if "ncdm" in dd else None,
            "ncdm_test_acc": dd["ncdm"]["test_acc"] if "ncdm" in dd else None,
            "ncdm_test_f1": dd["ncdm"]["test_f1"] if "ncdm" in dd else None,
            "ncdm_test_rmse": dd["ncdm"]["test_rmse"] if "ncdm" in dd else None,

            "mfcd_best_epoch": dd["mfcd"]["best_epoch"] if "mfcd" in dd else None,
            "mfcd_test_auc": dd["mfcd"]["test_auc"] if "mfcd" in dd else None,
            "mfcd_test_acc": dd["mfcd"]["test_acc"] if "mfcd" in dd else None,
            "mfcd_test_f1": dd["mfcd"]["test_f1"] if "mfcd" in dd else None,
            "mfcd_test_rmse": dd["mfcd"]["test_rmse"] if "mfcd" in dd else None,
        }

        if "ncdm" in dd and "mfcd" in dd:
            row["delta_auc"] = dd["mfcd"]["test_auc"] - dd["ncdm"]["test_auc"]
            row["delta_acc"] = dd["mfcd"]["test_acc"] - dd["ncdm"]["test_acc"]
            row["delta_f1"] = dd["mfcd"]["test_f1"] - dd["ncdm"]["test_f1"]
            row["delta_rmse"] = dd["mfcd"]["test_rmse"] - dd["ncdm"]["test_rmse"]
        else:
            row["delta_auc"] = None
            row["delta_acc"] = None
            row["delta_f1"] = None
            row["delta_rmse"] = None

        out.append(row)

    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Compare original NCDM and MFCD")

    parser.add_argument("--datasets", type=str, default="all")
    parser.add_argument("--models", type=str, default="ncdm,mfcd")
    parser.add_argument("--data-root", type=str, default=os.path.join(ROOT, "data"))
    parser.add_argument("--save-root", type=str, default=os.path.join(ROOT, "output", "ncdm_compare"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)

    parser.add_argument(
        "--select-metric",
        type=str,
        default="auc",
        choices=["auc", "acc", "f1", "rmse"],
        help="Metric used on validation set to choose the best epoch."
    )

    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--fluency-lambda", type=float, default=0.3)
    parser.add_argument("--flu-latent-dim", type=int, default=32)
    parser.add_argument("--rt-cap-ms", type=float, default=120000.0)

    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)
    ensure_dir(args.save_root)

    all_datasets = ["math1", "math2", "frcsub", "assist17", "assist09", "junyi"]
    all_models = ["ncdm", "mfcd"]

    if args.datasets.strip().lower() == "all":
        dataset_list = all_datasets
    else:
        dataset_list = [x.strip().lower() for x in args.datasets.split(",") if x.strip()]

    if args.models.strip().lower() == "all":
        model_list = all_models
    else:
        model_list = [x.strip().lower() for x in args.models.split(",") if x.strip()]

    summary = []

    for dataset_name in dataset_list:
        for model_type in model_list:
            try:
                metrics = run_one_model_one_dataset(args, dataset_name, model_type)
                summary.append(metrics)
            except Exception as e:
                print(f"\n[ERROR] dataset={dataset_name}, model={model_type}")
                print(repr(e))
                summary.append({
                    "dataset": dataset_name,
                    "model": model_type,
                    "error": repr(e),
                })

    save_json(os.path.join(args.save_root, "summary.json"), summary)

    comparison = build_comparison_table(summary)
    save_json(os.path.join(args.save_root, "comparison.json"), comparison)

    if len(comparison) > 0:
        pd.DataFrame(comparison).to_csv(
            os.path.join(args.save_root, "comparison.csv"),
            index=False
        )

    print("\n" + "=" * 90)
    print(f"Comparison Summary (best epoch selected by valid {args.select_metric})")
    print("=" * 90)

    def fmt6(x):
        if x is None:
            return "None"
        try:
            if isinstance(x, float) and np.isnan(x):
                return "nan"
        except Exception:
            pass
        return f"{float(x):.6f}"

    for row in comparison:
        print(
            f'{row["dataset"]}\n'
            f'  NCDM    -> epoch={row["ncdm_best_epoch"]}, '
            f'AUC={fmt6(row["ncdm_test_auc"])}, '
            f'ACC={fmt6(row["ncdm_test_acc"])}, '
            f'F1={fmt6(row["ncdm_test_f1"])}, '
            f'RMSE={fmt6(row["ncdm_test_rmse"])}\n'
            f'  MFCD    -> epoch={row["mfcd_best_epoch"]}, '
            f'AUC={fmt6(row["mfcd_test_auc"])}, '
            f'ACC={fmt6(row["mfcd_test_acc"])}, '
            f'F1={fmt6(row["mfcd_test_f1"])}, '
            f'RMSE={fmt6(row["mfcd_test_rmse"])}\n'
            f'  Delta   -> '
            f'AUC={fmt6(row["delta_auc"])}, '
            f'ACC={fmt6(row["delta_acc"])}, '
            f'F1={fmt6(row["delta_f1"])}, '
            f'RMSE={fmt6(row["delta_rmse"])}'
        )


if __name__ == "__main__":
    main()
