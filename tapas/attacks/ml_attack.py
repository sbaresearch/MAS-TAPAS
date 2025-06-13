from typing import List
from abc import abstractmethod
import numpy as np
import pandas as pd

from .base_classes import Attack
from ..threat_models import LabelInferenceThreatModel

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import StandardScaler
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC, SVR


def majority_voting(predictions_list):
    """ Performs mayority voting from a vector containing predictions from different classifiers. 
    
    Parameters
    ----------
    predictions_list : list
        List containing prediction vectors of different classifiers.
    Returns
    -------
    ensemble_result: list
        List containing predictions based on majority voting.
    
    """
    ensemble_result=[]
    for i in range(len(predictions_list[0])):
        predictions=[ val[i] for val in predictions_list]
        most_frequent_val = most_frequent(predictions)
        ensemble_result.append(most_frequent_val)
        
    return ensemble_result

def most_frequent(entry_list):
    """
    Counts more frequent element in a list.
    """
    return max(set(entry_list), key=entry_list.count) 

def average_results(predictions_list):
    """ Performs simple average from a vector containing predictions from different regression models. 
    
    Parameters
    ----------
    predictions_list : list
        List containing prediction vectors of different regression models.
    Returns
    -------
    ensemble_result: list
        List containing predictions based on simple average.
    
    """
    ensemble_result=[]
    for i in range(len(predictions_list[0])):
        predictions=[ val[i] for val in predictions_list]
        avg_val = np.mean(predictions)
        ensemble_result.append(avg_val)
        
    return ensemble_result           

CAT_ESTIMATORS = {
    'RF': lambda: RandomForestClassifier(random_state=42),
    'SVM': lambda: SVC(probability=True),
    'NB':  lambda: GaussianNB(),
    'KNN': lambda: KNeighborsClassifier(),
    'LR':  lambda: LogisticRegression(),
}

NUM_ESTIMATORS = {
    'LR': lambda: LinearRegression(),
    'SVR':lambda: SVR(kernel='rbf'),
    'MLP': lambda: MLPRegressor(solver='adam')
}

class MLAttack(Attack):
    """
    Performs attribute inference via ML classifiers on quasi-identifiers.
    Computes accuracy/F1 (categorical) or MAE/R²/MAPE (numeric).
    Operates over combinations of quasi-identifiers of fixed key_length.
    """

    def __init__(
        self,
        quasi_identifiers: List = None,
        estimator: str = 'ENS',
        categorical = True,
        label = None        
    ):
        self.quasi_identifiers = quasi_identifiers
        self._label = label or f"MLAttack({estimator})"
        self.trained = False
        self.categorical = categorical

        if estimator == 'ENS':
            self.estimators = {key: est() for key, est in (CAT_ESTIMATORS if categorical else NUM_ESTIMATORS).items()}
            self.ensemble_function = majority_voting if categorical else average_results
        elif estimator == 'Dummy':
            self.estimators = {estimator: DummyClassifier() if categorical else DummyRegressor()}
        else:
            assert estimator in NUM_ESTIMATORS.keys() or estimator in CAT_ESTIMATORS.keys(), \
                "Estimator does not exist"
            assert (estimator in NUM_ESTIMATORS.keys() and not categorical) or (estimator in CAT_ESTIMATORS.keys() and categorical) , \
                "Selected estimator is not appropriate for the sensitive attribute type (categorical vs numerical)"
            
            self.estimators = {estimator: (CAT_ESTIMATORS if categorical else NUM_ESTIMATORS)[estimator]()}

    def train(
        self,
        threat_model: LabelInferenceThreatModel = None,
        num_samples: int = None
    ):
        """
        num_samples: not used, leaft so it follows Attack interface
        """
        assert isinstance(threat_model, LabelInferenceThreatModel), \
            "Need LabelInferenceThreatModel (e.g. TargetedAIA)."
        
        self.threat_model = threat_model

        real_data = threat_model.atk_know_data.attacker_knowledge._get_data()

        if not self.quasi_identifiers:
            self.quasi_identifiers =  [col for col in real_data.description.columns if col != threat_model.sensitive_attribute]
        
        self.categorical_columns = [ col for col in self.quasi_identifiers if col in real_data.description.one_hot_cols]
        self.numerical_columns = [ col for col in self.quasi_identifiers if col not in self.categorical_columns]

        if self.categorical:
            self.categorical_columns += [threat_model.sensitive_attribute]
        else: 
            self.numerical_columns += [threat_model.sensitive_attribute]

            
        # Preprocess numerical columns
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # Preprocess categorical columns
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        
        # Build transformer with preprocessing pipelines.
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("numerical", num_transformer, self.numerical_columns),
                ("categorical", cat_transformer, self.categorical_columns),
            ],
            verbose_feature_names_out = False,
            remainder='passthrough'
        )

        
        train_data=pd.DataFrame(self.preprocessor.fit_transform(real_data.data))

        if self.categorical:
            mapping = {cat: i for i, cat in enumerate(self.preprocessor.named_transformers_['categorical'].named_steps['encoder'].categories_[-1])}
            if len(list(set(mapping.values()).difference(set(threat_model.attribute_values)))) !=0:
                raise ValueError(f'The preprocessor yield different mapping ({mapping}) than provided possible values ({threat_model.attribute_values})')


        train_data.columns= self.preprocessor.get_feature_names_out() 
    
        X_train = train_data[self.quasi_identifiers]
        y_train = train_data[threat_model.sensitive_attribute].astype(int)     

        for model_name, model in self.estimators.items():
            model.fit(X_train,y_train)
        
        self.trained = True


    def attack(self, datasets: List[pd.DataFrame]) -> List[int]:
        """
        For membership-style output: for each dataset, return best guess (majority vote) of target attribute.
        """
        assert self.trained, "Train before attacking."
        assert len(datasets[0]) == 1, 'Synthetic datasets should have one record each'

        df =  pd.concat([data.data for data in datasets])
        test_data=pd.DataFrame(self.preprocessor.transform(df),columns=self.preprocessor.get_feature_names_out()) 

        y_preds = []
        for name, model in self.estimators.items():
            y_pred_m = model.predict(test_data[self.quasi_identifiers])
            y_preds.append(y_pred_m)
        
        predictions = y_preds[0] if len(y_preds) == 1 else self.ensemble_function(y_preds)
        return predictions

    def attack_score(self, datasets: List[pd.DataFrame]) -> List[float]:
        assert self.trained, "Attack must first be trained."

        df =  pd.concat([data.data for data in datasets])
        test_data=pd.DataFrame(self.preprocessor.transform(df),columns=self.preprocessor.get_feature_names_out())
        
        y_scores = []
        for name, model in self.estimators.items():
            y_score_m = model.predict_proba(test_data[self.quasi_identifiers])
            y_scores.append(y_score_m)
        
        scores = y_scores[0][:,1] if len(y_scores) == 1 else self.ensemble_function(np.array(y_scores)[:,:,1])
        return scores
    
    @property
    def label(self):
        return self._label
        
