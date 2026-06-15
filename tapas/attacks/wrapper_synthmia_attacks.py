from tapas.attacks.base_classes import Attack
from tapas.datasets import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from .third_party import synth_mia
from .third_party.synth_mia.base import BaseAttacker
from .third_party.synth_mia.evaluation import AttackEvaluator


def _continuous_column_indices(description):
    """
    Indices, in the `as_numeric` encoded array, of the continuous (numerical)
    features. Mirrors the column accounting of
    `tapas.datasets.utils.encode_data`:
      - 'finite'                      -> one-hot block (len = #categories)
      - 'finite/ordered' (list repr)  -> single ordinal column (left as-is)
      - everything else (real/...)    -> single raw continuous column (scaled)
    """
    indices = []
    cidx = 0
    for cdict in description:
        d_type = cdict["type"]
        d_repr = cdict["representation"]
        if d_type == "finite":
            cidx += d_repr if isinstance(d_repr, int) else len(d_repr)
        elif d_type == "finite/ordered" and not isinstance(d_repr, int):
            cidx += 1
        else:
            indices.append(cidx)
            cidx += 1
    return indices


class SynthMiaTapasWrapper(Attack):
    def __init__(self, attacker_name: str, seed: int = None,
                 scale_numeric: bool = False, **kwargs):
        """
        Args:
            synth_mia_attacker: An instance of a Synth-MIA attacker
                                (e.g., DCR_Attacker)
            seed: Optional RNG seed. If set, numpy/torch/random are re-seeded
                  before computing attack scores so that stochastic attackers
                  produce reproducible results.
            scale_numeric: If True, standard-scale the continuous columns before
                  scoring.
            label: Name for the TAPAS report
        """
        self.seed = seed
        self.scale_numeric = scale_numeric
        self._cont_idx = None
        # Dynamically find and instantiate the attacker class
        try:
            attacker_class = getattr(synth_mia, attacker_name, None)
            self.attacker = attacker_class(**kwargs)
            #check is a valid attack or raise error
            assert isinstance(
                    self.attacker, 
                    BaseAttacker), "Attacker class not supported in synth_mia."
            
        except AttributeError:
            raise ValueError(f"Attacker '{attacker_name}' not found in synth_mia.attackers")

        # Set the TAPAS label
        self._label = f"SynthMIA_{attacker_name}"
        self.ref_data = None

    def train(self, threat_model):
        """
        No-Box attacks don't require training. 
        We store reference data here if the threat model provides it.
        """
        self.threat_model = threat_model
                 
        # If your threat model has auxiliary data, we store it as 'ref'
        if hasattr(self.threat_model.atk_know_data, 'aux_data'):
            self.ref_data = self.threat_model.atk_know_data.aux_data.as_numeric
        
        

    def attack_score(self, datasets: list[Dataset]) -> np.ndarray:
        """
        TAPAS calls this with one or more synthetic datasets (releases).
        When multiple releases are provided, scores are averaged across them,
        which reduces noise and produces a stronger membership signal.
        """
        X_test_clean = self.threat_model._target_records.as_numeric

        # Re-seed before scoring so stochastic attackers are reproducible.
        if self.seed is not None:
            import random
            random.seed(self.seed)
            np.random.seed(self.seed)
            try:
                import torch
                torch.manual_seed(self.seed)
            except ImportError:
                pass

        # Identify continuous columns once (only needed when scaling).
        if self.scale_numeric and self._cont_idx is None:
            self._cont_idx = _continuous_column_indices(
                self.threat_model._target_records.description
            )

        all_scores = []
        for ds in datasets:
            X_test, synth, ref = X_test_clean, ds.as_numeric, self.ref_data
            if self.scale_numeric and self._cont_idx:
                # Fit the scaler on the synthetic release, apply to all arrays.
                X_test, synth, ref = self._scale(X_test, synth, ref)
            scores = self.attacker._compute_attack_scores(
                X_test=X_test,
                synth=synth,
                ref=ref,
            )
            all_scores.append(scores)

        return np.mean(all_scores, axis=0)

    def _scale(self, X_test, synth, ref):
        """Standard-scale the continuous columns (scaler fit on `synth`)."""
        cont = self._cont_idx
        scaler = StandardScaler().fit(synth[:, cont])

        def transform(arr):
            if arr is None:
                return None
            arr = arr.copy()
            arr[:, cont] = scaler.transform(arr[:, cont])
            return arr

        return transform(X_test), transform(synth), transform(ref)

    def attack(self, datasets: list[Dataset]) -> np.ndarray:
        """Binary decision based on scores."""
       
        scores = self.attack_score(datasets)

        # Use median of scores as threshold 
        decision_threshold = np.median(scores)
        return (scores > decision_threshold).astype(int)
    
    @property
    def label(self):
        """
        A label to describe this attack in reports.

        """
        return self._label
        
       