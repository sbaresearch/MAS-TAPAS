# -------------------------------------------------------------------------
# This file contains code derived from the Synth-MIA project.
# Original Repository: https://github.com/joshward96/Synth-MIA
# Original License: Apache License 2.0
#
# MODIFIED BY Daniela Martinez from SBA Research on 20.05.2026:
# - Added a PCA preprocessing step to prevent prevent matrix errors.
# -------------------------------------------------------------------------

import numpy as np
from scipy import stats
from tqdm import tqdm
from ..base import BaseAttacker
from typing import Dict, Any, Tuple, List, Optional
from .bnaf.density_estimation import density_estimator_trainer, compute_log_p_x
from sklearn.decomposition import PCA
import torch
from sklearn.neighbors import KDTree
from numpy.linalg import matrix_rank

class GenLRA(BaseAttacker):
    """
    Shadow-Box Generative Likelihood Ratio Attack (Gen-LRA)
    """
   
    def __init__(self, k_nearest=200, estimation_method="kde", 
                 bw_method="silverman", epochs=100, save=False, multi=False, **kwargs):
        """Initialize Gen-LRA attacker.
        
        Args:
            k_nearest: Number of nearest neighbors to consider (default: 200)
            estimation_method: Density estimation method, "kde" or "bnaf" (default: "kde")
            bw_method: Bandwidth method for KDE (default: "silverman")
            epochs: Number of epochs for BNAF training (default: 100)
            save: Whether to save BNAF model (default: False)
            multi: If True, compute scores for multiple neighbor counts (default: False)
            **kwargs: Additional parameters
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Store parameters as attributes
        super().__init__(k_nearest=k_nearest, estimation_method=estimation_method,
                        bw_method=bw_method, epochs=epochs, save=save, multi=multi, **kwargs)
        self.name = "Gen-LRA" + ("-Multi" if multi else "")
        self._pca_objects = {}
       
    @staticmethod
    def _find_closest_k_points(X, point, k):
        tree = KDTree(X)
        return tree.query([point], k=k)[1][0]
   
    def _fit_estimator(self, fit_data: np.ndarray,key="default"):
        method = self.estimation_method
       
        if method == "kde":
            n_samples, n_features = fit_data.shape
            if n_features >1:
                # Reduce dimension to full-rank
                pca = PCA(n_components=0.99)
                fit_data = pca.fit_transform(fit_data)
                self._pca_objects[key] = pca
            else:
                self._pca_objects[key] = None
            return stats.gaussian_kde(fit_data.T, bw_method=self.bw_method)
        elif method == "bnaf":
            return density_estimator_trainer(fit_data, epochs=self.epochs, save=self.save)[1]
        else:
            raise ValueError(f"Unknown method: {method}. Choose 'bnaf' or 'kde'.")

    def _compute_density(self, X_test: np.ndarray, model_fit: Any,key="default") -> np.ndarray:
        method = self.estimation_method

        if method == "kde":
            pca = self._pca_objects.get(key)
            # Drop last softmax column if used in fit
            if pca is not None:
                X_test = pca.transform(X_test)
            return model_fit.evaluate(X_test.T)
        elif method == "bnaf":
            return np.exp(
                compute_log_p_x(model_fit, torch.as_tensor(X_test).float().to(self.device)).cpu().detach().numpy()
            )
        else:
            raise ValueError(f"Unknown method: {method}. Choose 'bnaf' or 'kde'.")
    def _select_optimal_bandwidth(self, data: np.ndarray, n_folds: int = 5,
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
            bandwidths = np.linspace(0.1 * scott_factor, 2.0 * scott_factor, 10)
       
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
   
    def _compute_attack_scores(
        self,
        X_test: np.ndarray,
        synth: np.ndarray,
        ref: Optional[np.ndarray] = None
    ):
        de_H0 = self._fit_estimator(ref,'ho')

        if not self.multi:
            # Standard Gen-LRA: return single array
            results = np.zeros(len(X_test))
            
            for i, test_point in enumerate(tqdm(X_test, desc="Processing Test dataset")):
                de_Ha = self._fit_estimator(np.vstack([ref, test_point]),'ha')
               
                # Get closest k points from synth to x*
                synth_test_indices = GenLRA._find_closest_k_points(synth, test_point, self.k_nearest)
                synth_test = synth[synth_test_indices]

                # Compute likelihoods
                likelihoods_H0 = self._compute_density(synth_test, de_H0,'ho')
                likelihoods_Ha = self._compute_density(synth_test, de_Ha,'ha')

                # Compute log-likelihood ratio
                results[i] = self._compute_log_likelihood_ratio(likelihoods_H0, likelihoods_Ha)
           
            return results
        
        else:
            # Multi Gen-LRA: return dictionary with scores for different neighbor counts
            results = {n: [] for n in range(1, len(synth))}
            
            for test_point in tqdm(X_test, desc="Processing Test dataset"):
                de_Ha = self._fit_estimator(np.vstack([ref, test_point]))
                
                # Get all points ordered by distance
                full_distance_indices = GenLRA._find_closest_k_points(synth, test_point, len(synth))
                
                # Compute likelihoods for all points
                likelihoods_H0 = self._compute_density(synth[full_distance_indices], de_H0)
                likelihoods_Ha = self._compute_density(synth[full_distance_indices], de_Ha)
                
                # Compute scores for different neighbor counts
                for n in range(1, len(synth)):
                    synth_indices = full_distance_indices[:-n]
                    ratio = self._compute_log_likelihood_ratio(
                        likelihoods_H0[synth_indices], 
                        likelihoods_Ha[synth_indices]
                    )
                    results[n].append(ratio)
            
            return {n: np.array(scores) for n, scores in results.items()}
