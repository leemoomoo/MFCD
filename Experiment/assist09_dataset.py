
from __future__ import annotations

import ast
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


def _parse_knowledge_code(x) -> List[int]:
    if pd.isna(x):
        return []

    s = str(x).strip()
    if not s:
        return []

    try:
        vals = ast.literal_eval(s)
        if isinstance(vals, int):
            vals = [vals]
        elif not isinstance(vals, (list, tuple)):
            vals = [int(vals)]
        out = []
        for v in vals:
            vv = int(v)
            if vv >= 1:
                out.append(vv - 1)
        return out
    except Exception:
        s = s.strip("[]")
        if not s:
            return []
        out = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            vv = int(part)
            if vv >= 1:
                out.append(vv - 1)
        return out


def load_item_q_info(item_csv_path: str):
    if not os.path.exists(item_csv_path):
        raise FileNotFoundError(f"File not found: {item_csv_path}")

    df = pd.read_csv(item_csv_path)
    required = ["item_id", "knowledge_code"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {item_csv_path}: {missing}")

    df["item_id"] = pd.to_numeric(df["item_id"], errors="coerce")
    df = df.dropna(subset=["item_id"])
    df["item_id"] = df["item_id"].astype(int)

    item2skills_raw: Dict[int, List[int]] = {}
    max_k = -1

    for _, row in df.iterrows():
        raw_item_id = int(row["item_id"])
        skills = _parse_knowledge_code(row["knowledge_code"])
        item2skills_raw[raw_item_id] = skills
        if len(skills) > 0:
            max_k = max(max_k, max(skills))

    knowledge_n = max_k + 1 if max_k >= 0 else 0
    return item2skills_raw, knowledge_n


def load_split_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = ["user_id", "item_id", "score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
    df["item_id"] = pd.to_numeric(df["item_id"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    df = df.dropna(subset=["user_id", "item_id", "score"])
    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)
    df["score"] = (df["score"].astype(float) > 0.5).astype(np.float32)

    return df


def split_train_valid_test(
    df: pd.DataFrame,
    seed: int = 2025,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.1,
):
    """Random split on interaction rows with fixed 8:1:1 ratio."""
    rng = np.random.RandomState(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)

    n = len(df)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    return (
        df.iloc[idx[:n_train]].reset_index(drop=True),
        df.iloc[idx[n_train:n_train + n_valid]].reset_index(drop=True),
        df.iloc[idx[n_train + n_valid:]].reset_index(drop=True),
    )


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


class Assist09Dataset(Dataset):
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


def build_assist09_splits_from_folder(
    data_dir: str,
    use_fluency: bool = True,
    alpha: float = 1.0,
    beta: float = 1.0,
    allow_user_oov: bool = True,
    seed: int = 2025,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.1,
    train_file: str = "train.csv",
    valid_file: str = "valid.csv",
    test_file: str = "test.csv",
    item_file: str = "item.csv",
):
    train_path = os.path.join(data_dir, train_file)
    valid_path = os.path.join(data_dir, valid_file)
    test_path = os.path.join(data_dir, test_file)
    item_path = os.path.join(data_dir, item_file)

    train_df = load_split_csv(train_path)
    valid_df = load_split_csv(valid_path)
    test_df = load_split_csv(test_path)

    all_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    train_df, valid_df, test_df = split_train_valid_test(
        all_df,
        seed=seed,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
    )

    item2skills_raw, knowledge_n = load_item_q_info(item_path)
    user_map, _ = build_maps_from_train(train_df, item2skills_raw)
    item_map = {raw_i: idx for idx, raw_i in enumerate(sorted(item2skills_raw.keys()))}

    train_stats = None
    if use_fluency:
        train_stats = build_train_fluency_stats(train_df, user_map, item2skills_raw)

    train_ds = Assist09Dataset(
        train_df, item2skills_raw, knowledge_n, user_map, item_map, "train",
        use_fluency=use_fluency, train_stats=None,
        alpha=alpha, beta=beta, allow_user_oov=allow_user_oov
    )
    valid_ds = Assist09Dataset(
        valid_df, item2skills_raw, knowledge_n, user_map, item_map, "valid",
        use_fluency=use_fluency, train_stats=train_stats,
        alpha=alpha, beta=beta, allow_user_oov=allow_user_oov
    )
    test_ds = Assist09Dataset(
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
