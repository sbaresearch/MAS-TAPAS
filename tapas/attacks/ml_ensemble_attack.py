from typing import List
from abc import abstractmethod
import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.preprocessing import LabelEncoder


from tapas.datasets.dataset import TabularDataset
from tapas.threat_models.aia import NoBoxThreatModelAIA, TargetedAIA

from .base_classes import Attack
from ..threat_models import LabelInferenceThreatModel

from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor

      

CAT_ESTIMATORS = {
    'RF': RandomForestClassifier(random_state=42),
    'NB':   GaussianNB(),
    'KNN': KNeighborsClassifier(),
    'LR':  LogisticRegression(max_iter=500),
    'XGB': XGBClassifier(n_estimators=100, learning_rate=0.1, eval_metric='logloss')
}

NUM_ESTIMATORS = {
    'XGB': XGBRegressor(n_estimators=100, learning_rate=0.1),
    'RF':  RandomForestRegressor(n_estimators=100),
    'ET':  ExtraTreesRegressor(n_estimators=100),
}


class MLInferenceAttack(Attack):
    """
    Performs attribute inference via ML classifiers on quasi-identifiers.
    Computes accuracy/F1 (categorical) or MAE/R²/MAPE (numeric).
    Operates over combinations of quasi-identifiers of fixed key_length.
    """

    def __init__(
        self,
        label = None        
    ):
        self._label = label or f"MLInferenceAttack"
        self.trained = False
        self.mem = {}

    def train(
        self,
        threat_model
    ):
        """
        Train the attack by fitting the preprocessor and preparing the data.
        """
        if not isinstance(threat_model, NoBoxThreatModelAIA):
            raise TypeError(
                f"GeneralizedCAPAttack requires NoBoxThreatModelAIA, "
                f"but received {type(threat_model).__name__}."
            )
        
        self.threat_model = threat_model
        self.categorical = (threat_model.sensitive_attribute_type == "finite")
        self.mem = {}

        if self.categorical:
            self.voter = VotingClassifier(
                estimators=list(CAT_ESTIMATORS.items()),
                voting='soft', n_jobs=-1
            )
            # The classes come from the threat model.
            self.attribute_values = list(threat_model.attribute_values or [])
            if not self.attribute_values:
                raise ValueError(
                    "MLInferenceAttack requires the threat model to define the "
                    "possible values of the sensitive attribute, but "
                    f"attribute_values is {threat_model.attribute_values!r}."
                )
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(np.asarray(self.attribute_values))
        else:
            self.voter = VotingRegressor(
                estimators=list(NUM_ESTIMATORS.items()),
                n_jobs=-1
            )

        self.trained = True

    def attack(self, datasets: List[pd.DataFrame]) -> List[int]:
        """
        For each dataset, return best guess (majority vote or average) of target attribute.
        """
        scores = self.attack_score(datasets)
        
        if self.categorical:
            if scores.ndim == 1:
                # Binary target: 1 if prob > 0.5, else 0
                indices = (scores > 0.5).astype(int)
            else:
                # Multi-class: Index of highest probability
                indices = np.argmax(scores, axis=1)

            return np.array([self.attribute_values[i] for i in indices])
        
        # Continuous: Scores are the predictions
        return scores

    def attack_score(self, datasets: List[pd.DataFrame], proba=True) -> List[float]:
        assert self.trained, "Attack must first be trained."

        
        target_record_x = self.threat_model.target_record.view(
            exclude_columns=[self.threat_model.sensitive_attribute]
        )
        
        final_scores = []     
           
        for i,dataset in enumerate(datasets):
            key = f"dataset{i}"
            if key in self.mem:
                voter = self.mem[key]
            else:
                # Instantiane new voter (Classifier or Regressor)
                voter = clone(self.voter)
                
                # Prepare synthetic training data from the current dataset
                X_syn = dataset.view(exclude_columns=[self.threat_model.sensitive_attribute])
                y_syn = dataset.view(columns=[self.threat_model.sensitive_attribute])
                
                if self.categorical:
                    values_syn = y_syn.data.values.ravel()
                    unknown = set(np.unique(values_syn)) - set(self.attribute_values)
                    if unknown:
                        raise ValueError(
                            f"The synthetic data contains values of "
                            f"'{self.threat_model.sensitive_attribute}' that the "
                            "threat model does not declare in attribute_values: "
                            f"{sorted(unknown)}."
                        )
                    y_syn = self.label_encoder.transform(values_syn)
                else:
                    y_syn = y_syn.data.values.ravel()
                
                # Train ensemble on synthetic data
                voter.fit(X_syn.as_numeric, y_syn)
                
                # Save fit model for further target records
                self.mem[key] = voter
                          
            if self.categorical:
                # Categorical: predict_proba only has a column per class present
                # in this release, so expand it to one entry per declared value,
                # in attribute_values order.
                probs_present = voter.predict_proba(target_record_x.as_numeric)[0]
                res = np.zeros((len(self.attribute_values),))
                for encoded_label, probability in zip(voter.classes_, probs_present):
                    value = self.label_encoder.inverse_transform([encoded_label])[0]
                    res[self.attribute_values.index(value)] = probability
                if len(self.attribute_values) == 2:
                    res = res[1]  # Return probability of the positive (second) class
            else:
                # Numerical: return the predicted continuous value
                prediction = voter.predict(target_record_x.as_numeric)
                res = float(np.ravel(prediction)[0])
            
            final_scores.append(res)
            
        return  np.array(final_scores) 
    
    @property
    def label(self):
        return self._label
        
