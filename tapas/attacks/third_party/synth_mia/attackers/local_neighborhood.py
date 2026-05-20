import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from ..base import BaseAttacker
from typing import Dict, Any, Tuple, List, Optional

class LocalNeighborhood(BaseAttacker):
    def __init__(self, radius=1.0, metric='euclidean', **kwargs):
        """Initialize Local Neighborhood attacker.
        
        Args:
            radius: Radius for neighborhood search (default: 1.0)
            metric: Distance metric (default: 'euclidean')
            **kwargs: Additional parameters
        """
        super().__init__(radius=radius, metric=metric, **kwargs)
        self.name = "Local Neighborhood"
        
    def _compute_attack_scores(self, X_test: np.ndarray, synth: np.ndarray, ref: Optional[np.ndarray] = None) -> np.ndarray:
        """
        For each row in X_test, calculate the proportion of points in synth within a sphere of radius r.

        Parameters:
        X_test (np.ndarray): Array of test points.
        synth (np.ndarray): Synthetic dataset.
        ref (np.ndarray): Reference dataset (not used in this attack).

        Returns:
        np.ndarray: Array of proportions for each test point.
        """
        # Initialize NearestNeighbors with the given metric
        nbrs = NearestNeighbors(radius=self.radius, metric=self.metric).fit(synth)

        # Find neighbors within the radius for all points in X_test
        neighbors_within_r = nbrs.radius_neighbors(X_test, return_distance=False)

        # Calculate proportions for all test points in one go
        scores = np.array([len(neighbors) / len(synth) if len(synth) > 0 else 0 for neighbors in neighbors_within_r])

        return scores
