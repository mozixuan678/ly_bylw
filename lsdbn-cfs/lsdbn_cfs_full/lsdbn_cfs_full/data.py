from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class ADNIData:
    signals: np.ndarray
    labels: np.ndarray
    label_names: Tuple[str, str]
    subjects: np.ndarray
    image_ids: np.ndarray
    raw_groups: np.ndarray


def _labels_to_binary(groups: np.ndarray) -> np.ndarray:
    labels = []
    for value in groups.astype(str):
        v = value.strip().upper()
        if v in {"CN", "NC", "NORMAL", "CONTROL"}:
            labels.append(0)
        elif "EMCI" in v:
            labels.append(1)
        else:
            raise ValueError("Unsupported label value: %r" % value)
    return np.asarray(labels, dtype=np.int64)


def _load_npz(path: Path) -> ADNIData:
    z = np.load(str(path), allow_pickle=True)
    signals = np.asarray(z["data"], dtype=np.float32)
    raw_groups = np.asarray(z["labels"]).astype(str)
    subjects = np.asarray(z["subjects"]).astype(str) if "subjects" in z else np.arange(len(raw_groups)).astype(str)
    image_ids = np.asarray(z["image_ids"]).astype(str) if "image_ids" in z else subjects
    return ADNIData(
        signals=signals,
        labels=_labels_to_binary(raw_groups),
        label_names=("CN", "EMCI"),
        subjects=subjects,
        image_ids=image_ids,
        raw_groups=raw_groups,
    )


def load_adni_data(path: str) -> ADNIData:
    """Load ADdata.npy or a version-stable ADdata_arrays.npz fallback."""
    data_path = Path(path)
    if data_path.suffix.lower() == ".npz":
        return _load_npz(data_path)

    try:
        obj = np.load(str(data_path), allow_pickle=True).item()
        signals = np.asarray(obj["data"], dtype=np.float32)
        label_table = obj["label"]
        raw_groups = label_table["Group"].astype(str).to_numpy()
        subjects = label_table["Subject"].astype(str).to_numpy() if "Subject" in label_table else np.arange(len(raw_groups)).astype(str)
        image_ids = label_table["Image Data ID"].astype(str).to_numpy() if "Image Data ID" in label_table else subjects
        return ADNIData(
            signals=signals,
            labels=_labels_to_binary(raw_groups),
            label_names=("CN", "EMCI"),
            subjects=subjects,
            image_ids=image_ids,
            raw_groups=raw_groups,
        )
    except Exception as exc:
        fallback = data_path.with_name("ADdata_arrays.npz")
        if fallback.exists():
            print("ADdata.npy could not be read by this Python environment; using %s" % fallback)
            return _load_npz(fallback)
        raise RuntimeError(
            "Failed to load %s. If this is a pandas pickle compatibility issue, run "
            "tools/convert_addata.py with an environment that can read ADdata.npy."
            % data_path
        ) from exc


def stratified_subject_split(
    labels: np.ndarray,
    random_state: int = 42,
    train_ratio: float = 0.60,
    fs_ratio: float = 0.30,
    test_ratio: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subject-level split with exact class-wise rounded counts.

    For 70 CN and 62 EMCI, this gives 42/21/7 and 37/19/6,
    i.e. 79 train, 40 feature-selection and 13 test subjects.
    """
    rng = np.random.RandomState(random_state)
    train_parts = []
    fs_parts = []
    test_parts = []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * train_ratio))
        n_fs = int(round(n * fs_ratio))
        n_train = min(max(n_train, 1), n - 2)
        n_fs = min(max(n_fs, 1), n - n_train - 1)
        train_parts.append(idx[:n_train])
        fs_parts.append(idx[n_train:n_train + n_fs])
        test_parts.append(idx[n_train + n_fs:])
    train = np.concatenate(train_parts)
    fs = np.concatenate(fs_parts)
    test = np.concatenate(test_parts)
    rng.shuffle(train)
    rng.shuffle(fs)
    rng.shuffle(test)
    return train.astype(np.int64), fs.astype(np.int64), test.astype(np.int64)


def n_sliding_windows(n_timepoints: int, window: int, step: int) -> int:
    if n_timepoints < window:
        raise ValueError("Window length is larger than the time-series length.")
    return (n_timepoints - window) // step + 1


def repeat_subject_labels(labels: np.ndarray, subject_indices: np.ndarray, n_windows: int) -> np.ndarray:
    return np.repeat(labels[subject_indices], n_windows).astype(np.int64)


def save_split_metadata(path: str, adni: ADNIData, split: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    train_idx, fs_idx, test_idx = split
    payload: Dict[str, object] = {
        "train_subject_indices": train_idx.tolist(),
        "feature_selection_subject_indices": fs_idx.tolist(),
        "test_subject_indices": test_idx.tolist(),
        "train_subject_ids": adni.subjects[train_idx].tolist(),
        "feature_selection_subject_ids": adni.subjects[fs_idx].tolist(),
        "test_subject_ids": adni.subjects[test_idx].tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

