import numpy as np
from sklearn.neighbors import NearestNeighbors
from ..base import BaseAttacker
from typing import Dict, Any, Tuple, List, Optional

class DPI(BaseAttacker):
    """
    Shadow-Box Data Plagiarism Index (DPI) attack.
    Ward, J., Wang, C.-H., and Cheng, G. Data plagiarism index: Characterizing the privacy risk of data-copying in tabular generative models. 
    KDD- Generative AI Evaluation Workshop, 2024. URL https://arxiv.org/abs/2406.13012.
    """
    def __init__(self, distance='l2', k_nearest=20, **kwargs):
        """Initialize DPI attacker.
        
        Args:
            distance: Distance metric (default: 'l2')
            k_nearest: Number of nearest neighbors (default: 20)
            **kwargs: Additional parameters
        """
        super().__init__(distance=distance, k_nearest=k_nearest, **kwargs)
        self.name = "DPI"

    @staticmethod         
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def _compute_attack_scores(
        self, 
        X_test: np.ndarray, 
        synth: np.ndarray, 
        ref: Optional[np.ndarray] = None
    ) -> np.ndarray:
        k = self.k_nearest
        
        # Concatenate ref and synth arrays
        nn_data = np.concatenate((ref, synth), axis=0)

        # Create a NearestNeighbors object and fit it to the data
        nn = NearestNeighbors(n_neighbors=k, metric=self.distance).fit(nn_data)
    
        # Find the k nearest neighbors for each element in test
        distances, indices = nn.kneighbors(X_test)
    
        # Compute the ratio of KNN from ref and synth
        scores = []
        for idx in indices:
            # Count the number of neighbors from ref and synth
            ref_count = np.sum(idx < len(ref))
            synth_count = k - ref_count
        
            # Compute the ratio
            score = DPI.sigmoid(synth_count / (ref_count + 1e-20))
            scores.append(score)
    
        return np.array(scores)
