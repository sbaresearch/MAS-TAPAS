from typing import List
from abc import abstractmethod
import numpy as np
import pandas as pd
from sklearn import clone


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
        categorical = True,
        label = None        
    ):
        self._label = label or f"MLInferenceAttack"
        self.trained = False
        self.categorical = categorical
        self.estimators = {key: est for key, est in (CAT_ESTIMATORS if categorical else NUM_ESTIMATORS).items()}
        self.voter = VotingClassifier(estimators=list(self.estimators.items()),voting='soft',n_jobs=-1) if self.categorical else VotingRegressor(estimators=list(self.estimators.items()),n_jobs=-1)
        self.mem = {}

    def train(
        self,
        threat_model: LabelInferenceThreatModel = None,
        num_samples: int = None
    ):
        """
        Train the attack by fitting the preprocessor and preparing the data.
        """
        assert isinstance(threat_model, TargetedAIA | NoBoxThreatModelAIA), \
             "Need LabelInferenceThreatModel (e.g. TargetedAIA)."
        
        self.threat_model = threat_model
        self.trained = True        

    def attack(self, datasets: List[pd.DataFrame]) -> List[int]:
        """
        For each dataset, return best guess (majority vote) of target attribute.
        """
        scores = self.attack_score(datasets)
        
        if self.categorical:
            if scores.ndim == 1:
                # Binary target: 1 if prob > 0.5, else 0
                indices = (scores > 0.5).astype(int)
            else:
                # Multi-class: Index of highest probability
                indices = np.argmax(scores, axis=1) 

            # Get a fitted model from memory
            voter = next(iter(self.mem.values()))

            predictions = voter.le_.inverse_transform(indices)
        
        else:
            predictions = scores           
            
        return predictions

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
                
                # Train ensemble on synthetic data
                voter.fit(X_syn.as_numeric, y_syn.data.values.ravel())
                
                # Save fit model for further target records
                self.mem[key] = voter
                          
            if self.categorical:
                # Categorical: predict_proba[1] for binary or full vector for multi-class
                res = voter.predict_proba(target_record_x.as_numeric)[0]
                if len(res) == 2:
                    res = res[1]  # Return probability of positive class
            else:
                # Numerical: return the predicted continuous value
                res = voter.predict(target_record_x.as_numeric)[0]
            
            final_scores.append(res)
            
        return  np.array(final_scores) 
    
    @property
    def label(self):
        return self._label
        
