"""
Threat Models for Attribute Inference Attacks.

Attribute Inference Attacks aim at inferring the value of a sensitive attribute
for a target user, given some known attributes and the synthetic data.

"""

# Type checking stuff
from __future__ import annotations
from typing import TYPE_CHECKING

from tapas.datasets.dataset import TabularDataset

if TYPE_CHECKING:
    from ..attacks import Attack  # for typing
    from ..datasets import TabularRecord
    from ..generators import Generator # for typing

from ..generators import NoBoxGenerator
from .base_classes import ThreatModel, TrainableThreatModel
from .attacker_knowledge import (
    AttackerKnowledgeOnData,
    AttackerKnowledgeOnGenerator,
    NoBoxKnowledge,
    AttackerKnowledgeWithLabel,
    LabelInferenceThreatModel,
)
from ..report import AIAttackSummary, BinaryAIAttackSummary

import numpy as np


class AIALabeller(AttackerKnowledgeWithLabel):
    """
    Replace a record in the private dataset with a given target record,
    and randomly set the value of a given sensitive attribute in that record.

    """

    def __init__(
        self,
        attacker_knowledge: AttackerKnowledgeOnData,
        target_records: TabularRecord,
        sensitive_attribute: str,
        attribute_values: list,
        distribution: list = None,
    ):
        """
        Wrap an AttackerKnowledgeOnData object by appending a record with
        randomized sensitive attribute

        Parameters
        ----------
        attacker_knowledge: AttackerKnowledgeOnData
            The data knowledge from which datasets are generated.
        target_records: Dataset
            The target records to add to the dataset with different sensitive
            attribute values. If this contains more than one record, the values
            for each record is sampled independently from all others.
        sensitive_attribute: str
            The name of the attribute to randomise.
        attribute_values: list
            All values that the attribute can take.
        distribution: list (None as default)
            Distribution from which to sample attribute values, a list of real
            numbers in [0,1] which sums to 1. By default (None), the uniform
            distribution is used.
        """
        self.attacker_knowledge = attacker_knowledge
        self.target_records = target_records
        self.sensitive_attribute = sensitive_attribute
        self.attribute_values = attribute_values
        self.distribution = distribution

    def generate_datasets_with_label(
        self, num_samples: int, training: bool = True
    ) -> tuple[list[Dataset], list[int]]:
        """
        Generate `num_samples` training or testing datasets with corresponding
        labels (arbitrary ints or bools).

        """
        # Generate the datasets from the attacker knowledge.
        datasets = self.attacker_knowledge.generate_datasets(num_samples, training)
        # Sample target attributes i.i.d. for each record and dataset.
        all_labels = list(
            np.random.choice(
                self.attribute_values,
                size=(num_samples, len(self.target_records)),
                replace=True,
                p=self.distribution,
            )
        )
        # Modify the records with all possible values, and save the resulting
        # records for efficiency purposes.
        modified_records = []
        for r in self.target_records:
            L = {}
            for value in self.attribute_values:
                r = r.copy()
                r.set_value(self.sensitive_attribute, value)
                L[value] = r
            modified_records.append(L)
        # For each dataset, remove random records and add the modified records.
        mod_datasets = []
        for ds, labels in zip(datasets, all_labels):
            # Remove random entries from the dataset, all at once (so as to avoid
            # removing records added from target_records).
            ds = ds.drop_records(
                np.random.choice(len(ds), size=len(self.target_records), replace=False)
            )
            # Add records one by one, with corresponding label.
            for r, v, mod_r in zip(self.target_records, labels, modified_records):
                ds.add_records(mod_r[v], in_place=True)
            mod_datasets.append(ds)
        # Convert labels to a 1-dimensional list if only one target record is given.
        if len(self.target_records) == 1:
            all_labels = [l[0] for l in all_labels]
        # Replace the records in each dataset, and return the labels.
        return mod_datasets, all_labels

    @property
    def label(self):
        return self.attacker_knowledge.label


class TargetedAIA(LabelInferenceThreatModel):
    """
    This threat model implements a MIA with arbitrary attacker knowledge on
    data and generator.

    """

    def __init__(
        self,
        attacker_knowledge_data: AttackerKnowledgeOnData,
        target_record: TabularDataset,
        sensitive_attribute: str,
        attribute_values: list,
        attacker_knowledge_generator: AttackerKnowledgeOnGenerator,
        distribution: list = None,
        memorise_datasets: bool = True,
        iterator_tracker: Callable[[list], Iterable] = None,
        num_concurrent: int = 1,
    ):
        LabelInferenceThreatModel.__init__(
            self,
            AIALabeller(
                attacker_knowledge_data,
                target_record,
                sensitive_attribute,
                attribute_values,
                distribution,
            ),
            attacker_knowledge_generator,
            memorise_datasets,
            iterator_tracker=iterator_tracker,
            num_labels=len(target_record),
            num_concurrent=num_concurrent,
        )
        self.sensitive_attribute = sensitive_attribute
        self.attribute_values = attribute_values
        self.distribution = distribution
        # See mia.py for the following bit of code.
        if self.multiple_label_mode:
            self._target_records = [r for r in target_record]
            self.set_label(0)
        else:
            self.target_record = target_record

    # Wrap the test method to output a AIAttackSummary.
    def _wrap_output(self, truth_labels, pred_labels, scores, attack):
        # If only two values are possible, use the binary valued report.
        # The second value is treated as the positive label.
        if len(self.attribute_values) == 2:
            ReportClass = BinaryAIAttackSummary
            kwargs = {"positive_value": self.attribute_values[1]}
        # Otherwise, we use the more general class.
        else:
            ReportClass = AIAttackSummary
            kwargs = {}

        if self.num_labels > 1:
            target_id = ",".join([rec.label for rec in self._target_records])
        else:
            target_id = self.target_record.label
        return ReportClass(
            truth_labels,
            pred_labels,
            scores,
            generator_info=self.atk_know_gen.label,
            attack_info=attack.label,
            dataset_info=self.atk_know_data.label,
            target_id=target_id,
            sensitive_attribute=self.sensitive_attribute,
            **kwargs
        )

    def set_label(self, label):
        """
        If the attack is performed against multiple targets, this sets the
        target record to use when outputting labels.

        """
        # See mia.py for the following bit of code.
        LabelInferenceThreatModel.set_label(self, label)
        self.target_record = self._target_records[label]

class NoBoxThreatModelAIA(ThreatModel):
    """
    Threat model for a no-box synthetic data scenario.
    
    Assumptions:
    - Attacker only has access to a synthetic dataset (or multiple synthetic datasets)
    - Attacker may have auxiliary data from the same distribution
    - Attacker knows quasi-identifiers from the attributes 
    - Attacker wants to infer a sensitive attribute from a given record in the real dataset
    """
    def __init__(
        self,
        target_records: TabularDataset,
        sensitive_attribute: str,
        quasi_identifiers: list,
        attacker_knowledge_data: AttackerKnowledgeOnData,
        attacker_knowledge_generator: AttackerKnowledgeOnGenerator,
        attribute_values: list = None,
        target_data: str = None
    ):
        """
        Parameters
        ----------
        target_records: Dataset
            The target records to add to the dataset with different sensitive
            attribute values. If this contains more than one record, the values
            for each record is sampled independently from all others.
        sensitive_attribute: str
            The name of the attribute to randomise.
        quasi_identifiers: list
            Subset of attribute known by the attacker.
        attribute_values: list
            All values that the attribute can take.
        attacker_knowledge_generator: AttackerKnowledgeOnGenerator
            The generator knowledge available for the attacker, restricted to No-Box case
            that only samples from synthetic outputs.
        attacker_knowledge_data: AttackerKnowledgeOnData
            Background data for the attacker, used to extract 'test_data' 
            for baseline control records.
        """
        # Assertions to ensure no-box integrity
        assert isinstance(attacker_knowledge_generator, NoBoxKnowledge), \
            "Attacker knowledge on generator must be NoBox for this specific privacy test."   
        assert isinstance(attacker_knowledge_generator.generator, NoBoxGenerator), \
            "Attacker generator must be a NoBoxGenerator"
        
        self.atk_know_data = attacker_knowledge_data
        self.atk_know_gen = attacker_knowledge_generator
        self.attribute_values = attribute_values       
        self.quasi_identifiers = quasi_identifiers
        self.sensitive_attribute = sensitive_attribute 
        self.relevant_cols = self.quasi_identifiers +[self.sensitive_attribute]
        
        
        # Store all possible targets
        self._target_records = [r for r in target_records.view(columns=self.relevant_cols)]
        self.num_labels = len(self._target_records)
        
        # Store targets in hold-out set
        test_data = getattr(self.atk_know_data, "test_data", None)
        self._control_records = []
        if test_data is not None:
            self._control_records = [r for r in test_data.view(columns=self.relevant_cols)]
        
        # Set the initial state to the first record
        self.set_label(0) 
        
        # Type of target attribute (Retrieved using schema from target records)
        self.sensitive_attribute_type = target_records.description.schema[target_records.description.columns.index(sensitive_attribute)]['type']
        
        self.target_data = None
        
    def set_label(self, label: int, group='target'):
        """Sets the active record the attack will see."""
        self.current_label_index = label
        if group == 'target':
            self.target_record = self._target_records[label]
        else:
            self.target_record = self._control_records[label]
      
    def _run_attack_loop(self, attack, synthetic_datasets, records, is_control=False):
        """Iterates through specific group of records."""
        truths, preds, scores = [], [], []
        group_name = 'control' if is_control else 'target'
        
        for i in range(len(records)):
            self.set_label(i, group=group_name)
            
            # Record the real sensitive value
            truths.append(self.target_record.data[self.sensitive_attribute].iloc[0])
            
            # The attack internally calls self.threat_model.target_record
            preds.extend(attack.attack(synthetic_datasets))
            
            # Only collect scores for the target group 
            #if not is_control:
            scores.extend(attack.attack_score(synthetic_datasets))
                
        return truths, preds, scores
    
    def test(self, attack: Attack):
        # Generate synthetic data (list of one or more releases).
        raw_synthetic_datasets = self.atk_know_gen.generate(None, training_mode=False)
        synthetic_datasets = [ds.view(columns=self.relevant_cols) for ds in raw_synthetic_datasets]
        
        # Run attack on training data
        all_truth, all_preds, all_scores = self._run_attack_loop(
            attack, synthetic_datasets, self._target_records, is_control=False
        )
        
        # Run attack on hold-out data
        control_truth, control_preds = [], []
        if self._control_records:
            control_truth, control_preds, _ = self._run_attack_loop(
                attack, synthetic_datasets, self._control_records, is_control=True
            )
        
        # Wrap output with both sets of data
        return self._wrap_output(
            all_truth, 
            all_preds, 
            all_scores, 
            attack, 
            control_labels=control_truth, 
            control_preds=control_preds
        )
     
    def _wrap_output(self, truth_labels, pred_labels, scores, attack, control_labels=None, control_preds=None):
        # If only two values are possible, use the binary valued report.
        # The second value is treated as the positive label.
        if len(self.attribute_values) == 2:
            ReportClass = BinaryAIAttackSummary
            kwargs = {"positive_value": self.attribute_values[1]}
        # Otherwise, we use the more general class for AIA.
        else:
            ReportClass = AIAttackSummary
            kwargs={}

        if self.num_labels > 1:
            target_id = ",".join([rec.label for rec in self._target_records])
        else:
            target_id = self.target_record.label
        return ReportClass(
            truth_labels,
            pred_labels,
            scores,
            generator_info=self.atk_know_gen.label,
            attack_info=attack.label,
            dataset_info='Auxiliary',
            target_id=target_id,
            sensitive_attribute=self.sensitive_attribute,
            quasi_identifiers=self.quasi_identifiers,
            **kwargs
        )
        
    
        
        
        
        
        
        
    
    
    
    
    