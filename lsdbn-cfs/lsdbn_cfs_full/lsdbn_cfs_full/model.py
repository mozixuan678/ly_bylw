from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import ExperimentConfig
from .feature_selection import fisher_score, jcfs, mutual_info_score, rank_from_scores, relief_score, select_features_by_method
from .rbm import LogSumSparseRBM


@dataclass
class LayerState:
    rbm: LogSumSparseRBM
    hidden_full: np.ndarray
    selected: np.ndarray
    rank: np.ndarray
    scores: Dict[str, np.ndarray]


class LSDBNCFS:
    def __init__(
        self,
        config: ExperimentConfig,
        use_sparse: bool = True,
        use_jcfs: bool = True,
        jcfs_method: str = "jcfs",
    ) -> None:
        self.config = config
        self.use_sparse = bool(use_sparse)
        self.use_jcfs = bool(use_jcfs)
        self.jcfs_method = jcfs_method
        self.scaler_ = StandardScaler()
        self.input_rank_: Optional[np.ndarray] = None
        self.input_selected_: Optional[np.ndarray] = None
        self.layers_: List[LayerState] = []
        self.classifier_ = None
        self.selected_features_: Optional[np.ndarray] = None

    def _regularization(self, layer_index: int) -> Tuple[float, float, float]:
        if not self.use_sparse:
            return 0.0, 0.0, 0.0
        kl = self.config.kl_lambda_first if layer_index == 1 else self.config.kl_lambda_other
        return self.config.weight_decay, self.config.logsum_lambda, kl

    def _make_classifier(self):
        if self.config.classifier == "svm":
            return SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
        if self.config.classifier == "random_forest":
            return RandomForestClassifier(
                n_estimators=200,
                random_state=self.config.random_state,
                class_weight="balanced",
                n_jobs=1,
            )
        if self.config.classifier == "extra_trees":
            return ExtraTreesClassifier(
                n_estimators=600,
                max_features="sqrt",
                random_state=self.config.random_state,
                class_weight="balanced",
                n_jobs=1,
            )
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=self.config.random_state,
        )

    def fit_representation(self, X: np.ndarray, y: np.ndarray):
        Xs = self.scaler_.fit_transform(X).astype(np.float32)
        method0 = self.config.input_prefilter_method.lower()
        if method0 == "fisher":
            s0 = fisher_score(Xs, y)
        elif method0 in {"mi", "mutual_info"}:
            s0 = mutual_info_score(Xs, y, random_state=self.config.random_state)
        else:
            s0 = relief_score(Xs, y, n_neighbors=self.config.relief_neighbors)
        self.input_rank_ = rank_from_scores(s0)
        k0 = min(self.config.input_prefilter_k, Xs.shape[1])
        self.input_selected_ = self.input_rank_[:k0].copy()
        current = Xs[:, self.input_selected_]
        self.layers_ = []

        for layer_num, (n_hidden, keep) in enumerate(zip(self.config.hidden_layers, self.config.keep_per_layer()), start=1):
            weight_decay, logsum_lambda, kl_lambda = self._regularization(layer_num)
            rbm = LogSumSparseRBM(
                n_visible=current.shape[1],
                n_hidden=n_hidden,
                visible_type="gaussian" if layer_num == 1 else "bernoulli",
                lr=self.config.lr_first if layer_num == 1 else self.config.lr_other,
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                cd_k=self.config.cd_k,
                weight_decay=weight_decay,
                logsum_lambda=logsum_lambda,
                kl_lambda=kl_lambda,
                sparsity_target=self.config.sparsity_target,
                logsum_eps=self.config.logsum_eps,
                random_state=self.config.random_state + layer_num,
                device=self.config.device,
            ).fit(current)
            hidden = rbm.transform(current)
            if self.use_jcfs:
                if self.jcfs_method == "jcfs":
                    selected, rank, scores = jcfs(
                        hidden,
                        y,
                        k=keep,
                        beta=self.config.jcfs_beta,
                        ecfs_alpha=self.config.ecfs_alpha,
                        relief_neighbors=self.config.relief_neighbors,
                        random_state=self.config.random_state + layer_num,
                    )
                else:
                    selected, rank, scores = select_features_by_method(
                        hidden,
                        y,
                        k=keep,
                        method=self.jcfs_method,
                        random_state=self.config.random_state + layer_num,
                        relief_neighbors=self.config.relief_neighbors,
                        ecfs_alpha=self.config.ecfs_alpha,
                    )
            else:
                rank = np.arange(hidden.shape[1], dtype=np.int64)
                selected = rank.copy()
                scores = {"all": np.ones(hidden.shape[1], dtype=np.float64)}
            self.layers_.append(LayerState(rbm=rbm, hidden_full=hidden, selected=selected, rank=rank, scores=scores))
            current = hidden[:, selected]
        return self

    def transform_top(self, X: np.ndarray) -> np.ndarray:
        if self.input_selected_ is None:
            raise RuntimeError("fit_representation must be called first.")
        current = self.scaler_.transform(X).astype(np.float32)[:, self.input_selected_]
        for layer in self.layers_:
            hidden = layer.rbm.transform(current)
            current = hidden[:, layer.selected]
        return current.astype(np.float32)

    def layer_activations(self, X: np.ndarray) -> List[np.ndarray]:
        """Return full hidden activations for every layer on a new dataset."""
        if self.input_selected_ is None:
            raise RuntimeError("fit_representation must be called first.")
        current = self.scaler_.transform(X).astype(np.float32)[:, self.input_selected_]
        activations: List[np.ndarray] = []
        for layer in self.layers_:
            hidden = layer.rbm.transform(current)
            activations.append(hidden)
            current = hidden[:, layer.selected]
        return activations

    def fit_classifier(self, X: np.ndarray, y: np.ndarray, selected_features: Optional[np.ndarray] = None):
        if selected_features is None:
            Z = self.transform_top(X)
        else:
            Z = self.scaler_.transform(X).astype(np.float32)[:, selected_features]
        self.classifier_ = self._make_classifier().fit(Z, y)
        return self

    def predict(self, X: np.ndarray, selected_features: Optional[np.ndarray] = None) -> np.ndarray:
        if self.classifier_ is None:
            raise RuntimeError("fit_classifier must be called first.")
        if selected_features is None:
            Z = self.transform_top(X)
        else:
            Z = self.scaler_.transform(X).astype(np.float32)[:, selected_features]
        return self.classifier_.predict(Z)

    def bcfs(
        self,
        y_labels: np.ndarray,
        kappa: Optional[float] = None,
        selected_k: Optional[int] = None,
        activations: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Backward causal feature selection, returning original 0-based feature indices."""
        if not self.layers_ or self.input_selected_ is None:
            raise RuntimeError("fit_representation must be called first.")
        kappa = self.config.kappa if kappa is None else float(kappa)
        selected_k = self.config.selected_k if selected_k is None else int(selected_k)
        if activations is None:
            activations = [layer.hidden_full for layer in self.layers_]
        if len(activations) != len(self.layers_):
            raise ValueError("activations must have one matrix per layer.")

        deltas: List[np.ndarray] = []
        classes = np.unique(y_labels)
        if len(classes) != 2:
            raise ValueError("BCFS expects a binary task.")
        for h in activations:
            d = np.abs(h[y_labels == classes[0]].mean(axis=0) - h[y_labels == classes[1]].mean(axis=0))
            deltas.append(d.astype(np.float64))

        top = self.layers_[-1]
        top_mask = np.zeros_like(deltas[-1])
        top_mask[top.selected] = 1.0
        contribution = deltas[-1] * top_mask

        for li in range(len(self.layers_) - 1, 0, -1):
            w = np.abs(self.layers_[li].rbm.W.detach().cpu().numpy()).astype(np.float64)
            q_in = w @ contribution
            prev = self.layers_[li - 1]
            q = np.zeros(prev.hidden_full.shape[1], dtype=np.float64)
            q[prev.selected] = q_in
            bft = np.flatnonzero(q > kappa)
            m = max(len(bft), self.config.m_min)
            fused = np.union1d(bft, prev.rank[:min(m, len(prev.rank))])
            contribution = np.zeros(prev.hidden_full.shape[1], dtype=np.float64)
            contribution[fused] = deltas[li - 1][fused]

        w1 = np.abs(self.layers_[0].rbm.W.detach().cpu().numpy()).astype(np.float64)
        q0 = w1 @ contribution
        bft0 = np.flatnonzero(q0 > kappa)
        m0 = max(len(bft0), self.config.m_min)
        local_rank = rank_from_scores(q0)
        fused0 = np.union1d(bft0, local_rank[:min(m0, len(local_rank))])

        if len(fused0) < selected_k:
            fused0 = np.union1d(fused0, local_rank[:min(selected_k, len(local_rank))])
        fused0_ranked = fused0[np.argsort(q0[fused0])[::-1]]
        fused0_ranked = fused0_ranked[:min(selected_k, len(fused0_ranked))]
        self.selected_features_ = self.input_selected_[fused0_ranked].astype(np.int64)
        return self.selected_features_
