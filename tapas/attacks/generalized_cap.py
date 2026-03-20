import numpy as np 
from sklearn.calibration import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors, RadiusNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tapas.datasets.dataset import Dataset
from tapas.attacks.base_classes import Attack

class GeneralizedCAPAttack(Attack):
    """
    Generalized Correct Attribution Probability (GCAP) Attack for categorical attributes. 
    Hittmeir, M., Mayer, R., & Ekelhart, A. (2020, March). A baseline for attribute disclosure risk in synthetic data. 
    In Proceedings of the Tenth ACM Conference on Data and Application Security and Privacy (pp. 133-143).
    
    The attack finds the minimal distance (rho) to define an equivalence class 
    N in the synthetic data S for a victim record j and using an ensemble to determine the sensitive attribute value.
    """
    def __init__(
        self,
        tolerance=0.01,
        label: str = None,
    ):
        self._label = label or f"GeneralizedCAPAttack"
        self.tolerance = tolerance
        self.preprocessor = None
        self.label_encoder = LabelEncoder()
        
    def _build_internal_preprocessor(self, df):
        """Preprocessing pipeline for GCAP."""
        
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', MinMaxScaler())
        ])

        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipe, num_cols),
                ("cat", cat_pipe, cat_cols),
            ],
            remainder='drop'
        )
        
        self.preprocessor.fit(df)   
    
    def train(self, threat_model):
        """Train parameters for the attack.

        Parameters
        ----------
        threat_model: NoBoxThreatModelAIA.
        The threat model from which to generate labelled samples of predicted sensitive values,
        simulating an attacker with only access to synthetic datasets.
        """
        self.threat_model = threat_model
        self.trained = True
        
    
    def attack_score(self, datasets: list[Dataset]):
        """
        Computes the probability distribution of the 
        sensitive attribute within the minimal distance radius (rho).
        
        Parameters
        ----------
        datasets: a list of synthetic datasets.

        Returns
        -------
        scores: array of size len(datasets).

        """
        
        all_scores = []
        
        # Initialize preprocessor for the dataset
        if self.preprocessor is None:
            self._build_internal_preprocessor(datasets[0].data[self.threat_model.quasi_identifiers])
            
        # Get the target record
        target_record_x = self.threat_model.target_record.view(
            exclude_columns=[self.threat_model.sensitive_attribute]
        )  
        x_target = self.preprocessor.transform(target_record_x.data)
        
        for dataset in datasets:
            # Use preprocessor to transform the data
            X_syn = self.preprocessor.transform(dataset.view(exclude_columns=[self.threat_model.sensitive_attribute]).data)
            y_syn = self.label_encoder.fit_transform(dataset.view(columns=[self.threat_model.sensitive_attribute]).data.values.ravel())
                  
            # Find Rho (Algorithm 3.1: while N is empty, increase r)
            finder = NearestNeighbors(metric='manhattan')
            finder.fit(X_syn)
            dist_to_closest, _ = finder.kneighbors(x_target, n_neighbors=1)
            rho = dist_to_closest[0][0]
            
            # Extract Equivalence Class (N)
            # Radius includes rho + tolerance to catch floating errors and close enough numerical values
            effective_radius = rho + self.tolerance + 1e-7
        
            # FRNN is used to find the equivalence class  
            neigh = RadiusNeighborsClassifier(
                radius=effective_radius, 
                metric='manhattan', 
            )
            neigh.fit(X_syn, y_syn)
            
            # Get probabilities for all possible sensitive values
            probs = neigh.predict_proba(x_target)[0]
            
            # If binary target, returns probability of positive class 
            if len(probs) == 2:
                all_scores.append(probs[1])
            else:
                all_scores.append(probs)
                 
        return np.array(all_scores)
    
    def attack(self, datasets):
        """ Make a prediction on the sensitive attribute for each dataset.

        Parameters
        ----------
        datasets: a list of synthetic datasets.

        Returns
        -------
        predictions: np.array of predicted sensitive values.
        
        """
        
        scores = self.attack_score(datasets)
        
        if scores.ndim == 1:
            # Binary target: 1 if prob > 0.5, else 0
            indices = (scores > 0.5).astype(int)
        else:
            # Multi-class: Index of highest probability
            indices = np.argmax(scores, axis=1)
            
        predictions = self.label_encoder.inverse_transform(indices)
        
        return predictions

    @property
    def label(self):
        return self._label