from typing import Optional, Tuple

from geomloss import SamplesLoss
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import multivariate_normal
import numpy as np

def compute_metrics_baseline(
    y_scores: np.ndarray, y_true: np.ndarray, sample_weight: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    y_pred = y_scores > np.median(y_scores)
    acc = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    auc = roc_auc_score(y_true, y_scores, sample_weight=sample_weight)
    return acc, auc

def compute_wd(
    X_syn: np.ndarray,
    X: np.ndarray,
) -> float:
    X_ = X.copy()
    X_syn_ = X_syn.copy()
    if len(X_) > len(X_syn_):
        X_syn_ = np.concatenate(
            [X_syn_, np.zeros((len(X_) - len(X_syn_), X_.shape[1]))]
        )

    scaler = MinMaxScaler().fit(X_)

    X_ = scaler.transform(X_)
    X_syn_ = scaler.transform(X_syn_)

    X_ten = torch.from_numpy(X_).reshape(-1, 1)
    Xsyn_ten = torch.from_numpy(X_syn_).reshape(-1, 1)
    OT_solver = SamplesLoss(loss="sinkhorn")

    return OT_solver(X_ten, Xsyn_ten).cpu().numpy().item()

class normal_func:
    def __init__(self, X: np.ndarray) -> None:
        self.var = np.ones_like(np.std(X, axis=0) ** 2)
        self.mean = np.zeros_like(np.mean(X, axis=0))

    def pdf(self, Z: np.ndarray) -> np.ndarray:
        return multivariate_normal.pdf(Z, self.mean, np.diag(self.var))
        # return multivariate_normal.pdf(Z, np.zeros_like(self.mean), np.diag(np.ones_like(self.var)))


class normal_func_feat:
    def __init__(
        self,
        X: np.ndarray,
        continuous: list,
    ) -> None:
        if np.any(np.array(continuous) > 1) or len(continuous) != X.shape[1]:
            raise ValueError("Continous variable needs to be boolean")
        self.feat = np.array(continuous).astype(bool)

        if np.sum(self.feat) == 0:
            raise ValueError("there needs to be at least one continuous feature")

        for i in np.arange(X.shape[1])[self.feat]:
            if len(np.unique(X[:, i])) < 10:
                print(f"Warning: feature {i} does not seem continous. CHECK")

        self.var = np.std(X[:, self.feat], axis=0) ** 2
        self.mean = np.mean(X[:, self.feat], axis=0)

    def pdf(self, Z: np.ndarray) -> np.ndarray:
        return multivariate_normal.pdf(Z[:, self.feat], self.mean, np.diag(self.var))

