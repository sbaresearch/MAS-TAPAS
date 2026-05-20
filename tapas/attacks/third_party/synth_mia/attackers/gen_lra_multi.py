import numpy as np
from scipy import stats
from tqdm import tqdm
from sklearn.decomposition import PCA
from ..base import BaseAttacker
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
from scipy import stats
from tqdm import tqdm
from ..base import BaseAttacker
from typing import Dict, Any, Tuple, List, Optional
from .bnaf.density_estimation import density_estimator_trainer, compute_log_p_x
import torch
from sklearn.neighbors import KDTree

class GenLRAMulti(BaseAttacker):
    """
    Shadow-Box Generative Likelihood Ratio Attack (Gen-LRA)
    """
   
    def __init__(self, hyper_parameters=None):
        default_params = {
            'k_nearest': 10,
            "estimation_method": "kde",
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "bnaf_params": {
                "epochs": 100,
                "save": False
            },
            "kde_params": {
                #"bw_method": "silverman"
            }
        }
        self.hyper_parameters = {**default_params, **(hyper_parameters or {})}
        super().__init__(self.hyper_parameters)    
        self.name = "Gen-LRA-Multi"
       
    @staticmethod
    def _find_closest_k_points(X, point, k):
        tree = KDTree(X)
        return tree.query([point], k=k)[1][0]
   
    def _fit_estimator(self, fit_data: np.ndarray):
        method = self.hyper_parameters["estimation_method"]
       
        estimators = {
            "kde": lambda data: stats.gaussian_kde(data.T,  **self.hyper_parameters["kde_params"]),
            "bnaf": lambda data: density_estimator_trainer(data, **self.hyper_parameters["bnaf_params"])[1]
        }

        if method not in estimators:
            raise ValueError(f"Unknown method: {method}. Choose 'bnaf' or 'kde'.")

        return estimators[method](fit_data)

    def _compute_density(self, X_test: np.ndarray, model_fit: Any) -> np.ndarray:
        method = self.hyper_parameters["estimation_method"]
        device = self.hyper_parameters["device"]

        density_methods = {
            "kde": lambda X: model_fit.evaluate(X.T),
            "bnaf": lambda X: np.exp(
                compute_log_p_x(model_fit, torch.as_tensor(X).float().to(device)).cpu().detach().numpy()
            )
        }

        if method not in density_methods:
            raise ValueError(f"Unknown method: {method}. Choose 'bnaf' or 'kde'.")

        return density_methods[method](X_test)
    def _select_optimal_bandwidth(self, data: np.ndarray, n_folds: int = 20,
                                bandwidths: Optional[List[float]] = None) -> float:
        """
        Select optimal bandwidth parameter for KDE using cross-validation.
       
        Args:
            data: Reference dataset to use for cross-validation
            n_folds: Number of cross-validation folds
            bandwidths: List of bandwidth values to try; if None, will use a range of values
       
        Returns:
            The bandwidth value with highest average log-likelihood
        """
        if bandwidths is None:
            # Create range of bandwidths to try (Scott's rule of thumb as reference)
            scott_factor = data.shape[0] ** (-1. / (data.shape[1] + 4))
            bandwidths = np.linspace(0.5 * scott_factor, 2.0 * scott_factor, 100)
        # Shuffle the data
        np.random.seed(42)
        shuffled_indices = np.random.permutation(len(data))
        data_shuffled = data[shuffled_indices]
       
        # Create folds
        fold_size = len(data) // n_folds
        scores = np.zeros((len(bandwidths), n_folds))
       
        for fold in range(n_folds):
            # Split data into training and validation
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else len(data)
            val_indices = np.arange(val_start, val_end)
            train_indices = np.setdiff1d(np.arange(len(data)), val_indices)
           
            X_train = data_shuffled[train_indices]
            X_val = data_shuffled[val_indices]
           
            # Evaluate each bandwidth
            for i, bw in enumerate(bandwidths):
                # Fit KDE with current bandwidth
                kde = stats.gaussian_kde(X_train.T, bw_method=bw)
               
                # Evaluate log-likelihood on validation set
                log_likelihood = np.sum(np.log(kde.evaluate(X_val.T) + 1e-20))
                scores[i, fold] = log_likelihood
       
        # Average scores across folds
        avg_scores = np.mean(scores, axis=1)
        best_idx = np.argmax(avg_scores)
        best_bandwidth = bandwidths[best_idx]
       
        return best_bandwidth
    def _compute_log_likelihood_ratio(self, likelihoods_H0, likelihoods_Ha):
        epsilon = 1e-20  # Small constant to avoid log(0)

        return np.sum(np.log(likelihoods_Ha+ epsilon)) - np.sum(np.log(likelihoods_H0+ epsilon))
    def _compute_attack_scores(self, X_test, synth, ref):
        """
        Compute attack scores using the gen_lra algorithm, with KDE and PCA fallback.
        """
        # if (self.hyper_parameters["estimation_method"] == "kde" and
        #     self.hyper_parameters["kde_params"]["bw_method"] == "silverman"):
        #     optimal_bw = self._select_optimal_bandwidth(ref)
        #     self.hyper_parameters["kde_params"]["bw_method"] = optimal_bw
        #     print(f"Selected optimal bandwidth: {optimal_bw}")
        #     kde_H0 = self._fit_estimator(ref)
        # Attempt initial KDE without PCA, apply PCA only if KDE fails
        kde_H0 = self._fit_estimator(ref)
        
        results = {n: [] for n in range(1, len(synth))}
        for test_point in tqdm(X_test, desc="Processing Test dataset"):            
            kde_Ha = self._fit_estimator(np.vstack([ref, test_point]))
            full_distance_indices = GenLRAMulti._find_closest_k_points(synth, test_point, len(synth))

            likelihoods_H0 = self._compute_density(synth[full_distance_indices], kde_H0)
            likelihoods_Ha = self._compute_density(synth[full_distance_indices], kde_Ha)
            for n in range(1, len(synth)):
                synth_indices = full_distance_indices[:-n]
            # Compute log-likelihood ratio
                results[n].append(self._compute_log_likelihood_ratio(likelihoods_H0[synth_indices], likelihoods_Ha[synth_indices]))
    
        return {n: np.array(scores) for n, scores in results.items()}
