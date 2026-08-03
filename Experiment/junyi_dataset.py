
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def pick_col(df: pd.DataFrame, candidates: List[str], name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find {name}. Tried {candidates}. Got columns={list(df.columns)[:80]}...")


@dataclass
class FluencyStats:
    attempt_cnt: Dict[Tuple[int, int], int]
    correct_cnt: Dict[Tuple[int, int], int]


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


def build_mappings_from_train(train_df: pd.DataFrame):
    user_col = pick_col(train_df, ["user_id", "user", "uid"], "user_id")
    item_col = pick_col(train_df, ["item_id", "problem_id", "exercise_id"], "item_id")
    skill_col = pick_col(train_df, ["skill_id", "skill", "kc", "knowledge_id"], "skill_id")

    users = sorted(pd.unique(train_df[user_col].astype(int)))
    items = sorted(pd.unique(train_df[item_col].astype(int)))
    skills = sorted(pd.unique(train_df[skill_col].astype(int)))

    user2idx = {int(u): i for i, u in enumerate(users)}
    item2idx = {int(it): i for i, it in enumerate(items)}
    skill2idx = {int(k): i for i, k in enumerate(skills)}
    return user2idx, item2idx, skill2idx


def compute_train_stats(
    train_df: pd.DataFrame,
    user2idx: Dict[int, int],
    skill2idx: Dict[int, int],
) -> FluencyStats:
    user_col = pick_col(train_df, ["user_id", "user", "uid"], "user_id")
    skill_col = pick_col(train_df, ["skill_id", "skill", "kc", "knowledge_id"], "skill_id")
    label_col = pick_col(train_df, ["correct", "score", "is_correct", "label"], "label")

    df = train_df.copy()
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df.dropna(subset=[label_col, user_col, skill_col])
    df[label_col] = (df[label_col].astype(float) > 0.5).astype(int)

    attempt_cnt: Dict[Tuple[int, int], int] = {}
    correct_cnt: Dict[Tuple[int, int], int] = {}

    u_raw = df[user_col].astype(int).values
    k_raw = df[skill_col].astype(int).values
    y = df[label_col].astype(int).values

    for u, k, yy in zip(u_raw, k_raw, y):
        if int(u) not in user2idx or int(k) not in skill2idx:
            continue
        ui = user2idx[int(u)]
        ki = skill2idx[int(k)]
        key = (ui, ki)
        attempt_cnt[key] = attempt_cnt.get(key, 0) + 1
        correct_cnt[key] = correct_cnt.get(key, 0) + int(yy)

    return FluencyStats(attempt_cnt=attempt_cnt, correct_cnt=correct_cnt)


class JunyiDataset(Dataset):
    """
    Unified output format:
      u, item, q_vec(one-hot), y, opp, acc_prior, stability
    """

    def __init__(
        self,
        df: pd.DataFrame,
        user2idx: Dict[int, int],
        item2idx: Dict[int, int],
        skill2idx: Dict[int, int],
        split: str,
        use_fluency: bool = False,
        train_stats: Optional[FluencyStats] = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        allow_user_oov: bool = True,
    ):
        assert split in {"train", "valid", "test"}
        self.split = split
        self.user2idx = user2idx
        self.item2idx = item2idx
        self.skill2idx = skill2idx
        self.K = len(skill2idx)

        user_col = pick_col(df, ["user_id", "user", "uid"], "user_id")
        item_col = pick_col(df, ["item_id", "problem_id", "exercise_id"], "item_id")
        skill_col = pick_col(df, ["skill_id", "skill", "kc", "knowledge_id"], "skill_id")
        label_col = pick_col(df, ["correct", "score", "is_correct", "label"], "label")

        d = df.copy()
        d[label_col] = pd.to_numeric(d[label_col], errors="coerce")
        d = d.dropna(subset=[label_col, user_col, item_col, skill_col])
        d[label_col] = (d[label_col].astype(float) > 0.5).astype(int)

        d = d[d[item_col].astype(int).isin(item2idx.keys())]
        d = d[d[skill_col].astype(int).isin(skill2idx.keys())]

        if allow_user_oov:
            self.user_oov_id = len(user2idx)
            self.student_n = len(user2idx) + 1
            u_idx = [user2idx.get(int(u), self.user_oov_id) for u in d[user_col].astype(int).values]
            self.users = np.array(u_idx, dtype=np.int64)
        else:
            missing_users = [int(x) for x in pd.unique(d[user_col].astype(int)) if int(x) not in user2idx]
            if missing_users:
                raise ValueError(f"[{split}] unseen users exist. Example={missing_users[:5]}")
            self.user_oov_id = None
            self.student_n = len(user2idx)
            self.users = d[user_col].astype(int).map(user2idx).astype(int).values

        self.items = d[item_col].astype(int).map(item2idx).astype(int).values
        self.skills = d[skill_col].astype(int).map(skill2idx).astype(int).values
        self.y = d[label_col].astype(np.float32).values

        self.exer_n = len(item2idx)
        self.knowledge_n = self.K

        n = len(self.y)
        self.opp = np.ones(n, dtype=np.float32)
        self.acc_prior = np.full(n, 0.5, dtype=np.float32)
        self.stability = np.full(n, 0.25, dtype=np.float32)

        if use_fluency:
            if split in {"valid", "test"}:
                if train_stats is None:
                    raise ValueError("valid/test requires train_stats.")

                for i, (ui, ki) in enumerate(zip(self.users, self.skills)):
                    if self.user_oov_id is not None and int(ui) == int(self.user_oov_id):
                        a, c = 0, 0
                    else:
                        a = train_stats.attempt_cnt.get((int(ui), int(ki)), 0)
                        c = train_stats.correct_cnt.get((int(ui), int(ki)), 0)

                    p = (c + alpha) / (a + alpha + beta)
                    self.opp[i] = float(a + 1)
                    self.acc_prior[i] = float(p)
                    self.stability[i] = float(p * (1.0 - p))

            else:
                attempt_cnt: Dict[Tuple[int, int], int] = {}
                correct_cnt: Dict[Tuple[int, int], int] = {}

                for i, (ui, ki, yy) in enumerate(zip(self.users, self.skills, self.y.astype(int))):
                    key = (int(ui), int(ki))
                    a = attempt_cnt.get(key, 0)
                    c = correct_cnt.get(key, 0)
                    p = (c + alpha) / (a + alpha + beta)

                    self.opp[i] = float(a + 1)
                    self.acc_prior[i] = float(p)
                    self.stability[i] = float(p * (1.0 - p))

                    attempt_cnt[key] = a + 1
                    correct_cnt[key] = c + int(yy)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        u = torch.tensor(self.users[idx], dtype=torch.long)
        it = torch.tensor(self.items[idx], dtype=torch.long)

        q_vec = torch.zeros(self.K, dtype=torch.float32)
        q_vec[int(self.skills[idx])] = 1.0

        y = torch.tensor(self.y[idx], dtype=torch.float32)
        opp = torch.tensor(self.opp[idx], dtype=torch.float32)
        acc_prior = torch.tensor(self.acc_prior[idx], dtype=torch.float32)
        stability = torch.tensor(self.stability[idx], dtype=torch.float32)

        return u, it, q_vec, y, opp, acc_prior, stability


def build_junyi_splits_from_csv(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    use_fluency: bool = False,
    alpha: float = 1.0,
    beta: float = 1.0,
    allow_user_oov: bool = True,
    seed: int = 2025,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.1,
):
    all_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    train_df, valid_df, test_df = split_train_valid_test(
        all_df,
        seed=seed,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
    )

    user2idx, item2idx, skill2idx = build_mappings_from_train(train_df)

    train_stats = None
    if use_fluency:
        train_stats = compute_train_stats(train_df, user2idx, skill2idx)

    train_ds = JunyiDataset(
        train_df, user2idx, item2idx, skill2idx, "train",
        use_fluency=use_fluency, train_stats=None,
        alpha=alpha, beta=beta, allow_user_oov=allow_user_oov
    )
    valid_ds = JunyiDataset(
        valid_df, user2idx, item2idx, skill2idx, "valid",
        use_fluency=use_fluency, train_stats=train_stats,
        alpha=alpha, beta=beta, allow_user_oov=allow_user_oov
    )
    test_ds = JunyiDataset(
        test_df, user2idx, item2idx, skill2idx, "test",
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
