
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class FluencyStats:
    attempt_cnt: Dict[Tuple[int, int], int]
    correct_cnt: Dict[Tuple[int, int], int]


def load_response_csv(csv_path: str) -> pd.DataFrame:
    """
    response.csv has no header.
    Expected 3 columns:
      col0 -> user_id
      col1 -> item_id
      col2 -> score
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path, header=None)

    if df.shape[1] < 3:
        raise ValueError(f"response.csv must have at least 3 columns, got {df.shape[1]}")

    df = df.iloc[:, :3].copy()
    df.columns = ["user_id", "item_id", "score"]

    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
    df["item_id"] = pd.to_numeric(df["item_id"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    df = df.dropna(subset=["user_id", "item_id", "score"]).copy()
    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)
    df["score"] = (df["score"].astype(float) > 0.5).astype(np.float32)

    return df


def load_q_matrix_csv(q_path: str):
    """
    q_matrix.csv has no header.
    Each row is an item, each column is a knowledge concept.
    Values are 0/1.
    Row index is treated as raw item_id.
    """
    if not os.path.exists(q_path):
        raise FileNotFoundError(f"File not found: {q_path}")

    q_df = pd.read_csv(q_path, header=None)
    q = q_df.values.astype(np.float32)

    if q.ndim != 2:
        raise ValueError(f"q_matrix must be 2D, got shape={q.shape}")

    knowledge_n = q.shape[1]
    item_n = q.shape[0]

    item2skills_raw: Dict[int, List[int]] = {}
    for raw_item_id in range(item_n):
        skills = np.where(q[raw_item_id] > 0.5)[0].tolist()
        item2skills_raw[int(raw_item_id)] = [int(k) for k in skills]

    return item2skills_raw, knowledge_n


def build_maps_from_train(train_df: pd.DataFrame, item2skills_raw: Dict[int, List[int]]):
    train_df = train_df[train_df["item_id"].isin(item2skills_raw.keys())].copy()

    user_ids = sorted(pd.unique(train_df["user_id"]))
    item_ids = sorted(pd.unique(train_df["item_id"]))

    user_map = {raw_u: idx for idx, raw_u in enumerate(user_ids)}
    item_map = {raw_i: idx for idx, raw_i in enumerate(item_ids)}
    return user_map, item_map


def build_train_fluency_stats(
    train_df: pd.DataFrame,
    user_map: Dict[int, int],
    item2skills_raw: Dict[int, List[int]],
):
    attempt_cnt: Dict[Tuple[int, int], int] = {}
    correct_cnt: Dict[Tuple[int, int], int] = {}

    for _, row in train_df.iterrows():
        raw_u = int(row["user_id"])
        raw_i = int(row["item_id"])
        y = 1 if float(row["score"]) > 0.5 else 0

        if raw_u not in user_map:
            continue
        if raw_i not in item2skills_raw:
            continue

        u_idx = user_map[raw_u]
        skills = item2skills_raw[raw_i]

        for k in skills:
            key = (u_idx, int(k))
            attempt_cnt[key] = attempt_cnt.get(key, 0) + 1
            correct_cnt[key] = correct_cnt.get(key, 0) + y

    return FluencyStats(attempt_cnt=attempt_cnt, correct_cnt=correct_cnt)


def split_train_valid_test(
    df: pd.DataFrame,
    seed: int = 2025,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.1,
):
    """
    Random split on interaction rows.
    """
    rng = np.random.RandomState(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)

    n = len(df)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train_idx = idx[:n_train]
    valid_idx = idx[n_train:n_train + n_valid]
    test_idx = idx[n_train + n_valid:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    valid_df = df.iloc[valid_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    return train_df, valid_df, test_df


class Assist17Dataset(Dataset):
    """
    Unified output:
      u, item, q_vec, y, opp, acc_prior, stability
    """

    def __init__(
        self,
        df: pd.DataFrame,
        item2skills_raw: Dict[int, List[int]],
        knowledge_n: int,
        user_map: Dict[int, int],
        item_map: Dict[int, int],
        split: str,
        use_fluency: bool = True,
        train_stats: Optional[FluencyStats] = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        allow_user_oov: bool = True,
    ):
        assert split in {"train", "valid", "test"}
        self.split = split
        self.knowledge_n = int(knowledge_n)
        self.exer_n = len(item_map)

        df = df[df["item_id"].isin(item_map.keys())].copy()

        if allow_user_oov:
            self.user_oov_id = len(user_map)
            self.student_n = len(user_map) + 1
            users_idx = [user_map.get(int(u), self.user_oov_id) for u in df["user_id"].values]
        else:
            df = df[df["user_id"].isin(user_map.keys())].copy()
            self.user_oov_id = None
            self.student_n = len(user_map)
            users_idx = [user_map[int(u)] for u in df["user_id"].values]

        self.users = np.array(users_idx, dtype=np.int64)
        self.items = np.array([item_map[int(x)] for x in df["item_id"].values], dtype=np.int64)
        self.raw_item_ids = np.array(df["item_id"].astype(int).values, dtype=np.int64)
        self.y = df["score"].astype(np.float32).values

        self.q_matrix = np.zeros((len(df), self.knowledge_n), dtype=np.float32)
        for idx, raw_item_id in enumerate(self.raw_item_ids):
            for k in item2skills_raw.get(int(raw_item_id), []):
                if 0 <= int(k) < self.knowledge_n:
                    self.q_matrix[idx, int(k)] = 1.0

        self.opp = np.ones(len(df), dtype=np.float32)
        self.acc_prior = np.full(len(df), 0.5, dtype=np.float32)
        self.stability = np.full(len(df), 0.25, dtype=np.float32)

        if use_fluency:
            if split in {"valid", "test"}:
                if train_stats is None:
                    raise ValueError("valid/test requires train_stats.")

                for i, (u_idx, raw_item_id) in enumerate(zip(self.users, self.raw_item_ids)):
                    skills = item2skills_raw.get(int(raw_item_id), [])
                    if len(skills) == 0:
                        continue

                    if self.user_oov_id is not None and int(u_idx) == int(self.user_oov_id):
                        self.opp[i] = 1.0
                        self.acc_prior[i] = 0.5
                        self.stability[i] = 0.25
                        continue

                    opps, priors, vars_ = [], [], []
                    for k in skills:
                        a = train_stats.attempt_cnt.get((int(u_idx), int(k)), 0)
                        c = train_stats.correct_cnt.get((int(u_idx), int(k)), 0)
                        p = (c + alpha) / (a + alpha + beta)

                        opps.append(a + 1.0)
                        priors.append(p)
                        vars_.append(p * (1.0 - p))

                    self.opp[i] = float(np.mean(opps))
                    self.acc_prior[i] = float(np.mean(priors))
                    self.stability[i] = float(np.mean(vars_))

            else:
                attempt_cnt: Dict[Tuple[int, int], int] = {}
                correct_cnt: Dict[Tuple[int, int], int] = {}

                for i, (u_idx, raw_item_id, yy) in enumerate(zip(self.users, self.raw_item_ids, self.y.astype(int))):
                    skills = item2skills_raw.get(int(raw_item_id), [])
                    if len(skills) == 0:
                        continue

                    opps, priors, vars_ = [], [], []
                    for k in skills:
                        key = (int(u_idx), int(k))
                        a = attempt_cnt.get(key, 0)
                        c = correct_cnt.get(key, 0)
                        p = (c + alpha) / (a + alpha + beta)

                        opps.append(a + 1.0)
                        priors.append(p)
                        vars_.append(p * (1.0 - p))

                    self.opp[i] = float(np.mean(opps))
                    self.acc_prior[i] = float(np.mean(priors))
                    self.stability[i] = float(np.mean(vars_))

                    for k in skills:
                        key = (int(u_idx), int(k))
                        attempt_cnt[key] = attempt_cnt.get(key, 0) + 1
                        correct_cnt[key] = correct_cnt.get(key, 0) + int(yy)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        u = torch.tensor(self.users[idx], dtype=torch.long)
        it = torch.tensor(self.items[idx], dtype=torch.long)
        q_vec = torch.tensor(self.q_matrix[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        opp = torch.tensor(self.opp[idx], dtype=torch.float32)
        acc_prior = torch.tensor(self.acc_prior[idx], dtype=torch.float32)
        stability = torch.tensor(self.stability[idx], dtype=torch.float32)

        return u, it, q_vec, y, opp, acc_prior, stability


def build_assist17_splits_from_folder(
    data_dir: str,
    use_fluency: bool = True,
    alpha: float = 1.0,
    beta: float = 1.0,
    allow_user_oov: bool = True,
    seed: int = 2025,
    response_file: str = "response.csv",
    q_file: str = "q_matrix.csv",
):
    response_path = os.path.join(data_dir, response_file)
    q_path = os.path.join(data_dir, q_file)

    df = load_response_csv(response_path)
    item2skills_raw, knowledge_n = load_q_matrix_csv(q_path)

    df = df[df["item_id"].isin(item2skills_raw.keys())].reset_index(drop=True)

    train_df, valid_df, test_df = split_train_valid_test(df, seed=seed)

    user_map, _ = build_maps_from_train(train_df, item2skills_raw)
    item_map = {raw_i: idx for idx, raw_i in enumerate(sorted(item2skills_raw.keys()))}

    train_stats = None
    if use_fluency:
        train_stats = build_train_fluency_stats(train_df, user_map, item2skills_raw)

    train_ds = Assist17Dataset(
        train_df, item2skills_raw, knowledge_n, user_map, item_map, "train",
        use_fluency=use_fluency, train_stats=None,
        alpha=alpha, beta=beta, allow_user_oov=allow_user_oov
    )
    valid_ds = Assist17Dataset(
        valid_df, item2skills_raw, knowledge_n, user_map, item_map, "valid",
        use_fluency=use_fluency, train_stats=train_stats,
        alpha=alpha, beta=beta, allow_user_oov=allow_user_oov
    )
    test_ds = Assist17Dataset(
        test_df, item2skills_raw, knowledge_n, user_map, item_map, "test",
        use_fluency=use_fluency, train_stats=train_stats,
        alpha=alpha, beta=beta, allow_user_oov=allow_user_oov
    )

    meta = {
        "student_n": train_ds.student_n,
        "exer_n": train_ds.exer_n,
        "knowledge_n": train_ds.knowledge_n,
        "n_train": len(train_ds),
        "n_valid": len(valid_ds),
        "n_test": len(test_ds),
    }
    return train_ds, valid_ds, test_ds, meta
