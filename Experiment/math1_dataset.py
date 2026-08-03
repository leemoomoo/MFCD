
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class FluencyStats:
    attempt_cnt: Dict[Tuple[int, int], int]
    correct_cnt: Dict[Tuple[int, int], int]


def _load_matrix_txt(path: str, dtype=np.float32) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    mat = np.loadtxt(path, dtype=dtype)
    if mat.ndim == 1:
        mat = mat[None, :]
    return mat


def _build_interactions(scores: np.ndarray, raw: Optional[np.ndarray], treat_zero_as_missing: bool):
    """
    Build interaction triplets from score matrix.

    Returns:
        users:  np.ndarray [N]
        items:  np.ndarray [N]
        labels: np.ndarray [N]
    """
    S, I = scores.shape
    interactions: List[Tuple[int, int, float]] = []

    for u in range(S):
        for it in range(I):
            y = float(scores[u, it])

            if treat_zero_as_missing:
                if raw is not None:
                    if float(raw[u, it]) == 0.0:
                        continue
                else:
                    if y == 0.0:
                        continue

            interactions.append((u, it, y))

    users = np.array([x[0] for x in interactions], dtype=np.int64)
    items = np.array([x[1] for x in interactions], dtype=np.int64)
    labels = np.array([1.0 if x[2] > 0.5 else 0.0 for x in interactions], dtype=np.float32)
    return users, items, labels


def _split_interactions(users, items, labels, seed=42, train_ratio=0.7, valid_ratio=0.1):
    """
    Random split on interaction rows.
    """
    n = len(labels)
    idx = np.arange(n)
    rng = np.random.RandomState(int(seed))
    rng.shuffle(idx)

    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    tr = idx[:n_train]
    va = idx[n_train:n_train + n_valid]
    te = idx[n_train + n_valid:]

    return (
        (users[tr], items[tr], labels[tr]),
        (users[va], items[va], labels[va]),
        (users[te], items[te], labels[te]),
    )


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


class Math1Dataset(Dataset):
    """
    Unified output format:
      u, item, q_vec, y, opp, acc_prior, stability

    Prefix-history logic:
      - train: online prefix-history construction
      - valid/test: use only TRAIN stats
    """

    def __init__(
        self,
        u: np.ndarray,
        i: np.ndarray,
        y: np.ndarray,
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

        self.u = u.astype(np.int64)
        self.i = i.astype(np.int64)
        self.y = y.astype(np.float32)
        self.q = q.astype(np.float32)

        self.student_n = int(student_n)
        self.exer_n = int(exer_n)
        self.knowledge_n = int(knowledge_n)

        n = len(self.y)
        self.opp = np.ones(n, dtype=np.float32)
        self.acc_prior = np.full(n, 0.5, dtype=np.float32)
        self.stability = np.full(n, 0.25, dtype=np.float32)

        if use_fluency:
            if split in {"valid", "test"}:
                if fluency_stats is None:
                    raise ValueError("valid/test requires fluency_stats built from TRAIN split.")
                self.opp, self.acc_prior, self.stability = build_fluency_proxies_from_stats(
                    self.u, self.i, self.q, fluency_stats, alpha=alpha, beta=beta
                )

            else:
                attempt_cnt: Dict[Tuple[int, int], int] = {}
                correct_cnt: Dict[Tuple[int, int], int] = {}

                for idx in range(n):
                    uu = int(self.u[idx])
                    ii = int(self.i[idx])
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

    def __getitem__(self, idx: int):
        u = torch.tensor(int(self.u[idx]), dtype=torch.long)
        it = torch.tensor(int(self.i[idx]), dtype=torch.long)
        q_vec = torch.tensor(self.q[int(self.i[idx])], dtype=torch.float32)
        y = torch.tensor(float(self.y[idx]), dtype=torch.float32)

        opp = torch.tensor(float(self.opp[idx]), dtype=torch.float32)
        acc_prior = torch.tensor(float(self.acc_prior[idx]), dtype=torch.float32)
        stability = torch.tensor(float(self.stability[idx]), dtype=torch.float32)

        return u, it, q_vec, y, opp, acc_prior, stability


def build_math1_splits_from_folder(
    data_dir: str,
    seed: int = 42,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.1,
    use_fluency: bool = False,
    alpha: float = 1.0,
    beta: float = 1.0,
    data_file: str = "data.txt",
    q_file: str = "q.txt",
    raw_file: str = "rawdata.txt",
    treat_zero_as_missing: bool = False,
):
    data_path = os.path.join(data_dir, data_file)
    q_path = os.path.join(data_dir, q_file)
    raw_path = os.path.join(data_dir, raw_file)

    scores = _load_matrix_txt(data_path, dtype=np.float32)
    q = _load_matrix_txt(q_path, dtype=np.float32)
    q = (q > 0.5).astype(np.float32)

    raw = _load_matrix_txt(raw_path, dtype=np.float32) if os.path.exists(raw_path) else None

    student_n, exer_n = scores.shape
    q_items, knowledge_n = q.shape
    if q_items != exer_n:
        raise ValueError(f"Q-matrix item count mismatch: data has I={exer_n}, q has I={q_items}")

    users, items, labels = _build_interactions(scores, raw, treat_zero_as_missing=treat_zero_as_missing)

    (u_tr, i_tr, y_tr), (u_va, i_va, y_va), (u_te, i_te, y_te) = _split_interactions(
        users, items, labels, seed=seed, train_ratio=train_ratio, valid_ratio=valid_ratio
    )

    fluency_stats = None
    if use_fluency:
        fluency_stats = build_train_fluency_stats(u_tr, i_tr, y_tr, q)

    train_ds = Math1Dataset(
        u_tr, i_tr, y_tr, q, student_n, exer_n, knowledge_n, "train",
        use_fluency=use_fluency, fluency_stats=None, alpha=alpha, beta=beta
    )
    valid_ds = Math1Dataset(
        u_va, i_va, y_va, q, student_n, exer_n, knowledge_n, "valid",
        use_fluency=use_fluency, fluency_stats=fluency_stats, alpha=alpha, beta=beta
    )
    test_ds = Math1Dataset(
        u_te, i_te, y_te, q, student_n, exer_n, knowledge_n, "test",
        use_fluency=use_fluency, fluency_stats=fluency_stats, alpha=alpha, beta=beta
    )

    meta = {
        "student_n": student_n,
        "exer_n": exer_n,
        "knowledge_n": knowledge_n,
        "n_train": len(train_ds),
        "n_valid": len(valid_ds),
        "n_test": len(test_ds),
    }
    return train_ds, valid_ds, test_ds, meta
