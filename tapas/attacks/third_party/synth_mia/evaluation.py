from typing import Dict, Any, List, Callable, Union, Tuple
import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    roc_curve, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    precision_recall_curve,
)
from scipy.stats import binomtest

class AttackEvaluator:
    """A comprehensive model evaluation toolkit with modular metrics and advanced privacy analysis.
    
    This class provides a unified interface for calculating various model performance 
    metrics, including classification metrics, ROC analysis, and privacy-focused evaluations.
    """
    def __init__(self, true_labels: np.ndarray, predicted_scores: np.ndarray):
        """Initialize the AttackEvaluator with ground truth labels and predicted scores.
        
        Args:
            true_labels: Ground truth binary labels.
            predicted_scores: Model's predicted probabilities or scores.
        """
        self.true_labels = np.asarray(true_labels)
        self.predicted_scores = np.asarray(predicted_scores)
        
    def _get_predicted_labels(self, decision_threshold: float = None) -> np.ndarray:
        """Convert predicted scores to binary labels using a threshold.
        
        Args:
            decision_threshold: Classification threshold. Defaults to median of predicted scores.
        
        Returns:
            np.ndarray: Binary predicted labels.
        """
        if decision_threshold is None:
            decision_threshold = np.median(self.predicted_scores)
        return (self.predicted_scores > decision_threshold).astype(int)
    
    def roc_metrics(self, target_fprs: List[float] = None) -> Dict[str, float]:
        """Calculate ROC-related metrics including AUC and TPR at specific FPRs.
        
        Args:
            target_fprs: False Positive Rate targets. Defaults to [0, 0.001, 0.01, 0.1].
        
        Returns:
            dict: ROC metrics including AUC and TPR at specified FPRs.
        """
        # Default FPR targets if not provided
        if target_fprs is None:
            target_fprs = [0, 0.001, 0.01, 0.1]
        
        # Calculate AUC
        metrics = {
            'auc_roc': roc_auc_score(self.true_labels, self.predicted_scores)
        }
        
        
        # Calculate TPR at specific FPRs
        fpr, tpr, _ = roc_curve(self.true_labels, self.predicted_scores)
        for target_fpr in target_fprs:
            metrics[f'tpr_at_fpr_{target_fpr}'] = np.interp(target_fpr, fpr, tpr)
        
        return metrics
    
    def classification_metrics(self, decision_threshold: float = None) -> Dict[str, float]:
        """Calculate standard binary classification metrics.
        
        Args:
            decision_threshold: Threshold for converting scores to labels. 
                Defaults to median of predicted scores.
        
        Returns:
            dict: Dictionary containing accuracy, precision, recall, F1 score, TPR, and FPR.
        """
        predicted_labels = self._get_predicted_labels(decision_threshold)
        cm = confusion_matrix(self.true_labels, predicted_labels)
        tn, fp, fn, tp = cm.ravel()

        return {
            "accuracy": accuracy_score(self.true_labels, predicted_labels),
            "precision": precision_score(self.true_labels, predicted_labels),
            "recall": recall_score(self.true_labels, predicted_labels),
            "f1_score": f1_score(self.true_labels, predicted_labels),
            "true_positive_rate": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
        }
    
    def privacy_metrics(self, decision_threshold: float = None) -> Dict[str, float]:
        """Calculate privacy-related metrics based on model predictions.
        
        Args:
            decision_threshold: Threshold for converting scores to labels. 
                Defaults to median of predicted scores.
        
        Returns:
            dict: Dictionary containing MIA advantage and privacy gain metrics.
        """
        predicted_labels = self._get_predicted_labels(decision_threshold)
        cm = confusion_matrix(self.true_labels, predicted_labels)
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        return {
            "mia_advantage": tpr - fpr,
            "privacy_gain": 1 - (tpr - fpr)
        }
    
    def epsilon_evaluator(
        self, 
        confidence_level: float = 0.9, 
        threshold_method: str = 'ratio', 
        validation_split: float = 0.1
    ) -> Dict[str, Union[float, Tuple[float, float]]]:
        """Evaluate the epsilon (privacy) bounds using Clopper-Pearson.
        
        Args:
            confidence_level: Confidence level for epsilon estimation. Defaults to 0.9.
            threshold_method: Method to select threshold ('ratio' or 'cp'). Defaults to 'ratio'.
            validation_split: Proportion of data for validation. Defaults to 0.1.
        
        Returns:
            dict: Dictionary containing:
                - threshold: Selected classification threshold
                - confidence_level: Confidence level used
                - epsilon_lower_bound: Lower bound of epsilon estimate
                - epsilon_upper_bound: Upper bound of epsilon estimate
                
        Raises:
            ValueError: If unknown thresholding method is specified.
        """
        # Shuffle data
        indices = np.arange(len(self.predicted_scores))
        np.random.shuffle(indices)

        # Shuffle both predicted_scores and true_labels using the same shuffled indices
        self.predicted_scores = self.predicted_scores[indices]
        self.true_labels = self.true_labels[indices]

        split_index = int(validation_split * len(self.predicted_scores))
        validation_scores = self.predicted_scores[split_index:]
        validation_labels =  self.true_labels[split_index:]
        
        # Select threshold based on method
        if threshold_method == 'cp':
            threshold = self._select_threshold_cp(
                self.predicted_scores[:split_index], 
                self.true_labels[:split_index], 
                confidence_level
            )
        elif threshold_method == 'ratio':
            threshold = self._select_threshold_ratio(
                self.predicted_scores[:split_index], 
                self.true_labels[:split_index], 
                confidence_level
            )
        else:
            raise ValueError(f"Unknown thresholding method {threshold_method}")
        
        # Estimate epsilon bounds
        eps_bounds = self._estimate_effective_epsilon_bounds(
            validation_scores, 
            validation_labels, 
            threshold, 
            confidence_level
        )
        
        return {
            "threshold": threshold,
            "confidence_level": confidence_level,
            "epsilon_lower_bound": eps_bounds[0],
            "epsilon_upper_bound": eps_bounds[1]
        }
    
    def _select_threshold_ratio(
        self, 
        scores: np.ndarray, 
        labels: np.ndarray, 
        conf_level: float = 0.9
    ) -> float:
        """Select an attack threshold using the ratio TP/FP method.
        
        Args:
            scores: Prediction scores.
            labels: True labels.
            conf_level: Confidence level. Defaults to 0.9.
        
        Returns:
            float: Selected threshold value.
        """
        best_eps = -1
        best_threshold = None
        positive_count = np.sum(labels == 1)
        negative_count = np.sum(labels == 0)

        # Adaptive minimum count calculations
        min_count = min(10, 1 + int(len(scores) * 0.1))
        min_count_for_num = max(10, int(len(scores) * 0.05))
        
        # Exclude first and last 10 thresholds
        unique_thresholds = np.unique(np.sort(scores)[min_count:-min_count])
        for threshold in unique_thresholds:
            true_positives = np.sum(scores[labels == 1] >= threshold)
            false_positives = np.sum(scores[labels == 0] >= threshold)
            
            # Compute TP/FP ratio
            num = true_positives / positive_count
            denom = false_positives / negative_count

            # Handle special cases for denominator
            if denom == 0:
                if num >= min_count_for_num:
                    best_eps = np.inf
                    min_count_for_num = num
                    best_threshold = threshold
            elif num / denom > best_eps:
                best_eps = num / denom
                best_threshold = threshold
        return best_threshold
    
    def _select_threshold_cp(
        self, 
        scores: np.ndarray, 
        labels: np.ndarray, 
        conf_level: float = 0.9
    ) -> float:
        """Select an attack threshold using the Clopper-Pearson method.
        
        Args:
            scores: Prediction scores.
            labels: True labels.
            conf_level: Confidence level. Defaults to 0.9.
        
        Returns:
            float: Selected threshold value.
        """
        best_eps = -1
        best_threshold = None
        
        for threshold in np.sort(np.unique(scores)):
            # Estimate effective epsilon bounds
            eps_bounds = self._estimate_effective_epsilon_bounds(
                scores, labels, threshold, conf_level
            )
            
            # Update best threshold if lower bound is higher
            eps_low = eps_bounds[0]
            if not np.isnan(eps_low) and eps_low > best_eps:
                best_eps = eps_low
                best_threshold = threshold

        return best_threshold
    
    def _estimate_effective_epsilon_bounds(
        self, 
        scores: np.ndarray, 
        labels: np.ndarray, 
        threshold: float, 
        confidence_level: float = 0.9
    ) -> Tuple[float, float]:
        """Estimate effective epsilon bounds using Clopper-Pearson confidence intervals.
        
        Args:
            scores: Prediction scores.
            labels: True labels.
            threshold: Classification threshold.
            confidence_level: Confidence level. Defaults to 0.9.
        
        Returns:
            tuple: Tuple containing:
                - float: Lower bound of epsilon estimate
                - float: Upper bound of epsilon estimate
        """
        confidence_level_half = 1 - (1 - confidence_level) / 2
        positive_label = 1.
        test = (lambda x: x >= threshold)
        # Compute positive and negative sample counts
        positive_count = np.sum(labels == positive_label)
        negative_count = np.sum(labels != positive_label)
        true_positives = np.sum(test(scores[labels == positive_label]))
        false_positives = np.sum(test(scores[labels != positive_label]))

        # Compute Clopper-Pearson confidence intervals
        bi_tpr = binomtest(
            k=true_positives, n=positive_count, p=true_positives / positive_count
        )
        ci_tpr = bi_tpr.proportion_ci(confidence_level_half)
        
        bi_fpr = binomtest(
            k=false_positives, n=negative_count, p=false_positives / negative_count
        )
        ci_fpr = bi_fpr.proportion_ci(confidence_level_half)
        
        # Calculate epsilon bounds
        low_bound = (
            max(0, np.log(ci_tpr.low / ci_fpr.high)) if ci_fpr.high > 0 else np.inf
        )
        high_bound = np.log(ci_tpr.high / ci_fpr.low) if ci_fpr.low > 0 else np.inf
        
        return low_bound, high_bound
