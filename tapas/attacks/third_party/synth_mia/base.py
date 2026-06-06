import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from .evaluation import AttackEvaluator

class BaseAttacker:
    """Base class for implementing a general attack model.

    This class serves as a base for attack models used in testing the privacy and security 
    of machine learning models. It provides methods for setting hyperparameters, executing attacks, 
    evaluating results, validating input data, and building consistent test data.
    """

    def __init__(self, **kwargs):
        """Initialize the attack model with hyperparameters.
        
        Args:
            **kwargs: Keyword arguments for hyperparameters specific to each attacker.
        """
        # Store parameters as instance attributes (sklearn style)
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Keep backward compatibility with hyper_parameters dict
        self.hyper_parameters = kwargs

    def attack(self, mem: np.ndarray, non_mem: np.ndarray, 
              synth: np.ndarray, ref: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Execute the attack given member, non-member, synthetic data, and optional reference data.

        Args:
            mem: Member data.
            non_mem: Non-member data.
            synth: Synthetic data.
            ref: Reference data. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: Predicted scores
                - np.ndarray: True labels
        """
        self._validate_input_data(mem, non_mem, synth, ref)
        
        X_test = self._build_X_test(mem, non_mem)
        y_true = self._build_y_test(mem, non_mem)

        # Implement the attack logic here
        scores = self._compute_attack_scores(X_test, synth, ref)

        #scores = np.where(np.isposinf(scores), 1e200, np.where(np.isneginf(scores), -1e200, scores))
        return y_true, scores

    def _compute_attack_scores(self, X_test: np.ndarray, 
                             synth: np.ndarray, 
                             ref: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute the attack scores. This method should be implemented by subclasses.

        Args:
            X_test: Test data (member and non-member).
            synth: Synthetic data.
            ref: Reference data. Defaults to None.

        Returns:
            np.ndarray: Predicted scores.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def eval(self, 
             true_labels: np.ndarray, 
             predicted_scores: np.ndarray, 
             metrics: List[str] = ["roc"], 
             **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate the attack using the ModelEvaluator class.
        
        Args:
            predicted_scores: Predicted scores from the attack.
            true_labels: True labels (0 for non-member, 1 for member).
            metrics: List of metrics to compute. Can include "roc", "classification", 
                "privacy", "epsilon". Defaults to ["roc"].
            **kwargs: Additional arguments for metrics computation:
                - target_fprs: List of target false positive rates for ROC metrics
                - decision_threshold: Threshold for classification/privacy metrics
                - confidence_level: Confidence level for epsilon evaluation
                - threshold_method: Method for threshold selection in epsilon evaluation
                - validation_split: Validation split ratio for epsilon evaluation
            
        Returns:
            dict: A dictionary of evaluation results for each metric.
        """
        evaluator = AttackEvaluator(true_labels, predicted_scores)
        results = {}
        
        if "roc" in metrics:
            target_fprs = kwargs.get("target_fprs", [0, 0.001, 0.01, 0.1])
            results.update(evaluator.roc_metrics(target_fprs=target_fprs))
        
        if "classification" in metrics:
            decision_threshold = kwargs.get("decision_threshold", None)
            results.update(evaluator.classification_metrics(decision_threshold=decision_threshold))
        
        if "privacy" in metrics:
            decision_threshold = kwargs.get("decision_threshold", None)
            results.update(evaluator.privacy_metrics(decision_threshold=decision_threshold))
        
        if "epsilon" in metrics:
            confidence_level = kwargs.get("confidence_level", 0.9)
            threshold_method = kwargs.get("threshold_method", "ratio")
            validation_split = kwargs.get("validation_split", 0.1)
            results.update(evaluator.epsilon_evaluator(
            confidence_level=confidence_level, 
            threshold_method=threshold_method, 
            validation_split=validation_split)
        )
        return results

    def get_properties(self) -> Dict[str, Any]:
        """Return the hyperparameters of the model.
        
        Returns:
            dict: The hyperparameters of the model.
        """
        return self.hyper_parameters

    def _validate_input_data(self, mem: np.ndarray, 
                           non_mem: np.ndarray, 
                           synth: np.ndarray, 
                           ref: Optional[np.ndarray] = None
    ) -> None:
        """Validate the input data to ensure consistency across numpy arrays.
        
        Args:
            mem: Member data.
            non_mem: Non-member data.
            synth: Synthetic data.
            ref: Reference data. Defaults to None.

        Raises:
            ValueError: If input arrays have inconsistent shapes or data types.
        """
        data_arrays = [mem, non_mem, synth]
        if ref is not None:
            data_arrays.append(ref)

        # Validate that all arrays have the same column count
        mem_columns = mem.shape[1]
        for array in data_arrays:
            if array.shape[1] != mem_columns:
                raise ValueError("All input arrays must have the same number of columns")

        # Validate that all arrays have the same data type
        mem_dtype = mem.dtype
        for array in data_arrays:
            if array.dtype != mem_dtype:
                raise ValueError("All input arrays must have the same data type")

    def _build_X_test(self, mem: np.ndarray, non_mem: np.ndarray) -> np.ndarray:
        """Build the X_test data by concatenating member and non-member data.
        
        Args:
            mem: Member data.
            non_mem: Non-member data.
        
        Returns:
            np.ndarray: The concatenated X_test data.
        """
        return np.concatenate([mem, non_mem], axis=0)
   
    def _build_y_test(self, mem: np.ndarray, non_mem: np.ndarray) -> np.ndarray:
        """Build the y_test data with labels for member and non-member data.
        
        Args:
            mem: Member data.
            non_mem: Non-member data.
        
        Returns:
            np.ndarray: The y_test labels (1 for members, 0 for non-members).
        """
        return np.concatenate([np.ones(len(mem)),np.zeros(len(non_mem))])
