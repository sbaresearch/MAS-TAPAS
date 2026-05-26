from tapas.attacks.base_classes import Attack
from tapas.datasets import Dataset
import numpy as np
from .third_party import synth_mia
from .third_party.synth_mia.base import BaseAttacker
from .third_party.synth_mia.evaluation import AttackEvaluator


class SynthMiaTapasWrapper(Attack):
    def __init__(self, attacker_name: str, **kwargs):
        """
        Args:
            synth_mia_attacker: An instance of a Synth-MIA attacker 
                                (e.g., DCR_Attacker)
            label: Name for the TAPAS report
        """
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
        TAPAS calls this with the synthetic dataset(s).
        """
        # Extract the synthetic data as a numpy array
        synth_data = datasets[0].as_numeric
        
        # Get the targets from the threat model context.
        X_test_clean = self.threat_model._target_records.as_numeric

        # Call Synth-MIA's engine directly
        scores = self.attacker._compute_attack_scores(
            X_test=X_test_clean,
            synth=synth_data,
            ref=self.ref_data
        )
        
        return scores

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
        
       