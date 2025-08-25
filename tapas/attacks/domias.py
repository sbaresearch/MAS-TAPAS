# future
from __future__ import absolute_import, division, print_function

# stdlib
from typing import Any, Dict, Optional

# third party
import numpy as np
import pandas as pd
import torch
from scipy import stats


# domias absolute
from .bnaf import compute_log_p_x, density_estimator_trainer, normal_func_feat
from .base_classes import Attack
from .utils import verbosed


class DOMIASAttack(Attack):
    def __init__(
        self,
        density_estimator: str = "prior",
        label = None,
        verbose = 0,
    ):
        
        if density_estimator.lower() not in ['prior', 'kde', 'bnaf']:
            raise ValueError(f"Density estimator should be one of the following: 'prior', 'kde', 'bnaf'")
        
        self.density_estimator = density_estimator.lower()
        self.trained = False
        self._label = label
        self.verbose = verbose

    @property
    def label(self):
        return  self._label if self._label else f"DOMIAS_{self.density_estimator}"
    
    def train(self, threat_model, num_samples = None):
        self.threat_model = threat_model
        
        self.reference_set = self.threat_model.atk_know_data.attacker_knowledge._get_data().as_numeric

        if self.density_estimator == "prior":
            # Fit a multivariate normal distribution to features
            continuous = []
            for i in np.arange(self.reference_set.shape[1]):
                if not np.issubdtype(self.reference_set[:, i].dtype, np.number) or len(np.unique((self.reference_set[:, i]))) < 3:
                    continuous.append(0)
                else:
                    continuous.append(1)

            self.norm = normal_func_feat(self.reference_set, continuous)
        elif self.density_estimator == "bnaf":
            # Train BNAF on the reference set
            _, self.p_R_model = density_estimator_trainer(
                    self.reference_set, 
                    epochs=50,
                    verbose=self.verbose)
        elif self.density_estimator == "kde":
            # Fit KDE on the reference set
            self.density_data = stats.gaussian_kde(self.reference_set.transpose(1, 0))
        
        self.trained = True

    def attack_score(self, datasets):
        assert self.trained, "Attack should be trained first"

        X_test = self.threat_model.target_record.as_numeric
        scores = []
        for dataset in datasets:
            synth_set = dataset.as_numeric
            verbosed(f"Shape of synth: {synth_set.shape}", self.verbose)
            if self.density_estimator == "bnaf":
                # Train BNAF on the synthetic set
                _, p_G_model = density_estimator_trainer(
                    synth_set,
                    verbose=self.verbose
                )
                p_G_evaluated = np.exp(
                    compute_log_p_x(p_G_model, torch.as_tensor(X_test).float())
                    .cpu()
                    .detach()
                    .numpy()
                )
                p_R_evaluated = np.exp(
                    compute_log_p_x(self.p_R_model, torch.as_tensor(X_test).float())
                    .cpu()
                    .detach()
                    .numpy()
                )
            elif self.density_estimator == "kde":
                # Fit KDE on the synthetic set
                density_gen = stats.gaussian_kde(synth_set.transpose(1, 0))
                p_G_evaluated = density_gen(X_test.transpose(1, 0))
                p_R_evaluated = self.density_data(X_test.transpose(1, 0))
            elif self.density_estimator == "prior":
                # Fit KDE on the synthetic set
                density_gen = stats.gaussian_kde(synth_set.transpose(1, 0))
                # density_data = stats.gaussian_kde(self.reference_set.transpose(1, 0))
                p_G_evaluated = density_gen(X_test.transpose(1, 0))
                p_R_evaluated = self.norm.pdf(X_test)

            # eqn2: \prop P_G(x_i)/P_X(x_i)
            scores.append(p_G_evaluated / (p_R_evaluated + 1e-10))

        return scores

    def attack(self, datasets):
        scores = self.attack_score(datasets)
        # Threshold at median to decide membership
        return [score > np.median(scores) for score in scores]
