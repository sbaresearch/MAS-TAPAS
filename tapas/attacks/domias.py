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
from .bnaf import compute_log_p_x, compute_wd, density_estimator_trainer, normal_func_feat
from .base_classes import Attack


class DOMIASAttack(Attack):
    def __init__(
        self,
        density_estimator: str = "prior",
    ):
        self.density_estimator = density_estimator
        self.trained = False

    @property
    def label(self):
        return "DOMIAS"
    
    def _convert_coorectly_to_numpy(self,df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        other_cols = df.select_dtypes(exclude=[np.number]).columns

        # Convert to numpy column by column and then stack
        arrays = []
        for col in df.columns:
            if col in numeric_cols:
                arrays.append(df[col].to_numpy())  # keep as numeric dtype
            else:
                arrays.append(df[col].to_numpy(dtype=object))  # store as objects
        
        return np.column_stack(arrays)
    
    def train(self, threat_model, num_samples = None):
        self.threat_model = threat_model
        
        self.reference_set = self.threat_model.atk_know_data.attacker_knowledge._get_data().as_numeric

        if self.density_estimator == "prior":
            continuous = []
            for i in np.arange(self.reference_set.shape[1]):
                if not np.issubdtype(self.reference_set[:, i].dtype, np.number) or len(np.unique((self.reference_set[:, i]))) < 3:
                    continuous.append(0)
                else:
                    continuous.append(1)

            self.norm = normal_func_feat(self.reference_set, continuous)
        
        self.trained = True


    def attack_score(self, datasets):
        assert self.trained, "Attack should be trained first"

        X_test = self.threat_model.target_record.as_numeric
        scores = []
        for dataset in datasets:
            synth_set = dataset.as_numeric
            if self.density_estimator == "bnaf":
                _, p_G_model = density_estimator_trainer(
                    synth_set,
                    epochs=2,
                    batch_dim=100,
                )
                _, p_R_model = density_estimator_trainer(self.reference_set, epochs=2, batch_dim=100)
                p_G_evaluated = np.exp(
                    compute_log_p_x(p_G_model, torch.as_tensor(X_test).float())
                    .cpu()
                    .detach()
                    .numpy()
                )
                # eqn2: \prop P_G(x_i)/P_X(x_i)
                # DOMIAS (BNAF for p_R estimation)
                p_R_evaluated = np.exp(
                    compute_log_p_x(p_R_model, torch.as_tensor(X_test).float())
                    .cpu()
                    .detach()
                    .numpy()
                )

            # KDE for pG
            elif self.density_estimator == "kde":
                density_gen = stats.gaussian_kde(synth_set.transpose(1, 0))
                density_data = stats.gaussian_kde(self.reference_set.transpose(1, 0))
                p_G_evaluated = density_gen(X_test.transpose(1, 0))
                # eqn2: \prop P_G(x_i)/P_X(x_i)
                # DOMIAS (BNAF for p_R estimation)
                p_R_evaluated = density_data(X_test.transpose(1, 0))
            elif self.density_estimator == "prior":
                density_gen = stats.gaussian_kde(synth_set.transpose(1, 0))
                density_data = stats.gaussian_kde(self.reference_set.transpose(1, 0))
                p_G_evaluated = density_gen(X_test.transpose(1, 0))
                # eqn2: \prop P_G(x_i)/P_X(x_i)
                # DOMIAS (BNAF for p_R estimation)
                p_R_evaluated = self.norm.pdf(X_test)

            p_rel = p_G_evaluated / (p_R_evaluated + 1e-10)

            print(p_rel)

            scores.append(p_rel)

        return scores

    def attack(self, datasets):
        scores = self.attack_score(datasets)
        # Threshold at 1 to decide membership
        return [score > 0.5 for score in scores]
