
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def read_q_matrix(q_path: str) -> np.ndarray:
    if not os.path.exists(q_path):
        raise FileNotFoundError(f"File not found: {q_path}")
    q = pd.read_csv(q_path, sep=r"\s+|\t+|,", engine="python", header=None).values.astype(np.float32)
    q = (q > 0.5).astype(np.float32)
    return q


def read_raw_interactions(raw_path: str, item_n: int):
    """
    Read rawdata.txt and build interactions.
    Supports missing = NaN or -1
    Label: score > 0 => 1 else 0
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"File not found: {raw_path}")

    raw_df = pd.read_csv(raw_path, sep=r"\s+|\t+|,", engine="python", header=None)
    user_n = raw_df.shape[0]
    if raw_df.shape[1] != item_n:
        raise ValueError("rawdata.txt columns != q.txt items")

    score_matrix = raw_df.values.astype(float)

    mask_nan = ~np.isnan(score_matrix)
    mask_neg = score_matrix >= 0
    mask = mask_nan & mask_neg

    u, it = np.where(mask)
    y = (score_matrix[u, it] > 0).astype(np.float32)

    records = pd.DataFrame({
        "user_id": u.astype(np.int64),
        "item_id": it.astype(np.int64),
        "score": y.astype(np.float32),
    })
    return records, user_n


@dataclass
class FluencyStats:
    attempt_cnt: Dict[Tuple[int, int], int]
    correct_cnt: Dict[Tuple[int, int], int]


def build_train_fluency_stats(u_arr: np.ndarray, i_arr: np.ndarray, y_arr: np.ndarray, q: np.ndarray) -> FluencyStats:
    """
    Build full TRAIN statistics.
    Used only for valid/test proxy construction.
    """
    attempt_cnt: Dict[Tuple[int, int], int] = {}
    correct_cnt: Dict[Tuple[int, int], int] = {}

    for uu, ii, yy in zip(u_arr, i_arr, y_arr):
        skills = np.where(q[int(ii)] > 0)[0]
        if skills.size == 0:
            continue

        is_corr = 1 if float(yy) > 0.5 else 0
        for kk in skills:
            key = (int(uu), int(kk))
            attempt_cnt[key] = attempt_cnt.get(key, 0) + 1
            correct_cnt[key] = correct_cnt.get(key, 0) + is_corr

    return FluencyStats(attempt_cnt=attempt_cnt, correct_cnt=correct_cnt)


def build_fluency_proxies_from_stats(
    u_arr: np.ndarray,
    i_arr: np.ndarray,
    q: np.ndarray,
    stats: FluencyStats,
    alpha: float = 1.0,
    beta: float = 1.0,
):
    """
    Build fluency proxies from precomputed stats.
    Used for valid/test, where only TRAIN stats should be used.
    """
    n = len(u_arr)
    opp = np.ones(n, dtype=np.float32)
    acc_prior = np.full(n, 0.5, dtype=np.float32)
    stability = np.full(n, 0.25, dtype=np.float32)

    for idx in range(n):
        uu = int(u_arr[idx])
        ii = int(i_arr[idx])
        skills = np.where(q[ii] > 0)[0]

        if skills.size == 0:
            continue

        opps, priors, vars_ = [], [], []
        for kk in skills:
            a = stats.attempt_cnt.get((uu, int(kk)), 0)
            c = stats.correct_cnt.get((uu, int(kk)), 0)
            p = (c + alpha) / (a + alpha + beta)

            opps.append(a + 1.0)
            priors.append(p)
            vars_.append(p * (1.0 - p))

        opp[idx] = float(np.mean(opps))
        acc_prior[idx] = float(np.mean(priors))
        stability[idx] = float(np.mean(vars_))

    return opp, acc_prior, stability


class Math2Dataset(Dataset):
    """
    Unified output format for MFCD:
      u, item, q_vec, y, opp, acc_prior, stability

    Prefix-history logic:
      - train: online prefix-history construction
      - valid/test: use only TRAIN stats
    """

    def __init__(
        self,
        df: pd.DataFrame,
        q: np.ndarray,
        student_n: int,
        exer_n: int,
        knowledge_n: int,
        split: str,
        use_fluency: bool = False,
        fluency_stats: Optional[FluencyStats] = None,
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        assert split in {"train", "valid", "test"}
        self.split = split

        self.u = df["user_id"].values.astype(np.int64)
        self.it = df["item_id"].values.astype(np.int64)
        self.y = df["score"].values.astype(np.float32)

        self.q = q.astype(np.float32)
        self.student_n = int(student_n)
        self.exer_n = int(exer_n)
        self.knowledge_n = int(knowledge_n)

        self.knowledge = np.zeros((len(self.it), self.knowledge_n), dtype=np.float32)
        for idx in range(len(self.it)):
            self.knowledge[idx] = self.q[int(self.it[idx])]

        n = len(self.y)
        self.opp = np.ones(n, dtype=np.float32)
        self.acc_prior = np.full(n, 0.5, dtype=np.float32)
        self.stability = np.full(n, 0.25, dtype=np.float32)

        if use_fluency:
            if split in {"valid", "test"}:
                if fluency_stats is None:
                    raise ValueError("valid/test requires fluency_stats built from TRAIN split.")
                self.opp, self.acc_prior, self.stability = build_fluency_proxies_from_stats(
                    self.u, self.it, self.q, fluency_stats, alpha=alpha, beta=beta
                )

            else:
                attempt_cnt: Dict[Tuple[int, int], int] = {}
                correct_cnt: Dict[Tuple[int, int], int] = {}

                for idx in range(n):
                    uu = int(self.u[idx])
                    ii = int(self.it[idx])
                    yy = int(self.y[idx] > 0.5)

                    skills = np.where(self.q[ii] > 0)[0]
                    if skills.size == 0:
                        continue

                    opps, priors, vars_ = [], [], []
                    for kk in skills:
                        key = (uu, int(kk))
                        a = attempt_cnt.get(key, 0)
                        c = correct_cnt.get(key, 0)
                        p = (c + alpha) / (a + alpha + beta)

                        opps.append(a + 1.0)
                        priors.append(p)
                        vars_.append(p * (1.0 - p))

                    self.opp[idx] = float(np.mean(opps))
                    self.acc_prior[idx] = float(np.mean(priors))
                    self.stability[idx] = float(np.mean(vars_))

                    for kk in skills:
                        key = (uu, int(kk))
                        attempt_cnt[key] = attempt_cnt.get(key, 0) + 1
                        correct_cnt[key] = correct_cnt.get(key, 0) + yy

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        u = torch.tensor(int(self.u[idx]), dtype=torch.long)
        it = torch.tensor(int(self.it[idx]), dtype=torch.long)
        kvec = torch.tensor(self.knowledge[idx], dtype=torch.float32)
        y = torch.tensor(float(self.y[idx]), dtype=torch.float32)

        opp = torch.tensor(float(self.opp[idx]), dtype=torch.float32)
        acc_prior = torch.tensor(float(self.acc_prior[idx]), dtype=torch.float32)
        stability = torch.tensor(float(self.stability[idx]), dtype=torch.float32)

        return u, it, kvec, y, opp, acc_prior, stability


def build_math2_splits_from_folder(
    data_dir: str,
    seed: int = 2025,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.1,
    use_fluency: bool = False,
    alpha: float = 1.0,
    beta: float = 1.0,
):
    q_path = os.path.join(data_dir, "q.txt")
    raw_path = os.path.join(data_dir, "rawdata.txt")

    q = read_q_matrix(q_path)
    item_n = int(q.shape[0])
    knowledge_n = int(q.shape[1])

    records, user_n = read_raw_interactions(raw_path, item_n=item_n)

    train_df, temp_df = train_test_split(
        records, test_size=(1 - train_ratio), random_state=seed, shuffle=True
    )

    valid_portion_in_temp = valid_ratio / (1 - train_ratio)
    valid_df, test_df = train_test_split(
        temp_df, test_size=(1 - valid_portion_in_temp), random_state=seed, shuffle=True
    )

    fluency_stats = None
    if use_fluency:
        fluency_stats = build_train_fluency_stats(
            train_df["user_id"].values,
            train_df["item_id"].values,
            train_df["score"].values,
            q
        )

    train_ds = Math2Dataset(
        train_df, q, user_n, item_n, knowledge_n, "train",
        use_fluency=use_fluency, fluency_stats=None, alpha=alpha, beta=beta
    )
    valid_ds = Math2Dataset(
        valid_df, q, user_n, item_n, knowledge_n, "valid",
        use_fluency=use_fluency, fluency_stats=fluency_stats, alpha=alpha, beta=beta
    )
    test_ds = Math2Dataset(
        test_df, q, user_n, item_n, knowledge_n, "test",
        use_fluency=use_fluency, fluency_stats=fluency_stats, alpha=alpha, beta=beta
    )

    meta = {
        "student_n": user_n,
        "exer_n": item_n,
        "knowledge_n": knowledge_n,
        "n_train": len(train_ds),
        "n_valid": len(valid_ds),
        "n_test": len(test_ds),
    }
    return train_ds, valid_ds, test_ds, meta
