from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ExperimentConfig:
    data_path: str = "../ADdata.npy"
    output_dir: str = "outputs/quick_run"
    random_state: int = 42
    repeats: int = 1
    methods: str = "all"
    force_recompute_dec: bool = False

    # Data and dEC settings.
    window: int = 25
    step: int = 5
    te_lag: int = 1
    te_eps: float = 1e-8
    train_ratio: float = 0.60
    fs_ratio: float = 0.30
    test_ratio: float = 0.10

    # Feature dimensions and selected feature count.
    input_prefilter_k: int = 512
    input_prefilter_method: str = "relief"
    selected_k: int = 124
    relief_neighbors: int = 1

    # LSDBN settings.
    hidden_layers: Tuple[int, ...] = (96, 48, 24)
    jcfs_keep: Tuple[int, ...] = (64, 32, 16)
    epochs: int = 3
    batch_size: int = 128
    cd_k: int = 1
    lr_first: float = 1e-3
    lr_other: float = 1e-2
    weight_decay: float = 5e-3
    logsum_lambda: float = 5e-3
    kl_lambda_first: float = 1e-3
    kl_lambda_other: float = 1e-4
    sparsity_target: float = 1e-2
    logsum_eps: float = 1e-3
    device: str = "auto"

    # JCFS and BCFS settings.
    jcfs_beta: Tuple[float, float, float] = (1.0 / 3, 1.0 / 3, 1.0 / 3)
    ecfs_alpha: float = 0.5
    kappa: float = 0.03
    kappa_values: Tuple[float, ...] = (0.0, 0.01, 0.03, 0.05, 0.08, 0.10)
    m_min: int = 20

    # Classifier.
    classifier: str = "logistic"  # logistic, svm, random_forest, extra_trees

    def keep_per_layer(self) -> Tuple[int, ...]:
        if len(self.jcfs_keep) == len(self.hidden_layers):
            return self.jcfs_keep
        return tuple(max(1, h // 2) for h in self.hidden_layers)


def preset_config(name: str) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if name == "paper":
        cfg.input_prefilter_k = 8010
        cfg.hidden_layers = (3000, 1800, 900)
        cfg.jcfs_keep = (1500, 900, 450)
        cfg.epochs = 50
        cfg.batch_size = 64
        cfg.m_min = 50
        cfg.repeats = 1
    elif name == "quick":
        cfg.input_prefilter_k = 512
        cfg.hidden_layers = (96, 48, 24)
        cfg.jcfs_keep = (64, 32, 16)
        cfg.epochs = 3
        cfg.batch_size = 128
        cfg.m_min = 20
        cfg.repeats = 1
    else:
        raise ValueError("Unknown preset: %s" % name)
    return cfg
