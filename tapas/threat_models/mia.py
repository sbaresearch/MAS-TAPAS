"""
Threat models for Membership Inference Attacks (MIA).

Membership inference attacks aim at detecting the presence of a specific
record in the training dataset from the synthetic dataset observed.

"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..attacks import Attack
    from ..datasets import Dataset
    from ..generators import Generator

from .base_classes import ThreatModel, TrainableThreatModel
from .attacker_knowledge import (
    AttackerKnowledgeOnData,
    AttackerKnowledgeOnGenerator,
    AttackerKnowledgeWithLabel,
    LabelInferenceThreatModel,
)
from ..report import MIAttackSummary

import numpy as np
import warnings

class MIA:
    """
    Minimal interface for Membership Inference Attacks.
    Any threat model that wants to run MIA attacks can inherit from this.
    """

    @property
    def target_record(self):
        """
        The record that the attack tries to infer membership for.
        Must be provided by any subclass.
        """
        raise NotImplementedError

class MIALabeller(AttackerKnowledgeWithLabel):
    """
    Randomly add a given target to the datasets sampled from auxiliary data.
    This class can be used to augment AttackerKnowledgeOnData objects that
    represent "generic" knowledge of the private dataset in order to use
    them for membership inference attacks.

    You may use this explicitly to feed into a LabelInferenceThreatModel, but
    this is meant mostly as an internal method to make MIAs.

    """

    def __init__(
        self,
        attacker_knowledge: AttackerKnowledgeOnData,
        target_records: Dataset,
        generate_pairs=True,
        replace_target=False,
    ):
        """
        Wrap an AttackerKnowledgeOnData object by appending a record.

        Parameters
        -----
        attacker_knowledge: AttackerKnowledgeOnData
            The data knowledge from which datasets are generated.
        target_records: Dataset
            The target records to append to the dataset. If several records
            are provided, these records are randomly added to the dataset
            independently from each other.
        generate_pairs: bool, default True
            Whether to output pairs of datasets differing only by the presence
            of the target record, or randomly choose for each dataset.
            If multiple targets are provided, then the pairs of datasets differ
            by exactly all of the multiple targets (as in, if a record x from
            the targets is in D, it is not in D', but the membership of each
            target is independent from the other targets).
        replace_target: bool, default False
            Whether to replace a record, instead of appending.

        """
        self.attacker_knowledge = attacker_knowledge
        self.target_records = target_records
        self.generate_pairs = generate_pairs
        self.replace_target = replace_target

    def generate_datasets_with_label(
        self, num_samples: int, training: bool = True
    ) -> tuple[list[Dataset], list[int]]:
        """
        Generate `num_samples` training or testing datasets with corresponding
        labels (arbitrary ints or bools).

        """
        # If generating pairs, make num_samples dividable by 2.
        if self.generate_pairs and num_samples // 2:
            num_samples += 1
        # Generate the datasets from the attacker knowledge.
        datasets = self.attacker_knowledge.generate_datasets(
            num_samples // 2 if self.generate_pairs else num_samples, training
        )
        # Compute modified datasets and corresponding labels by adding records
        # according to the labels. If self.generate_pairs, each iteration of
        # the loop creates two paired datasets.
        mod_datasets = []
        mod_labels = []
        for i_ds, dataset in enumerate(datasets):
            # Copy the datasets to be modified in place.
            dataset = dataset.copy()
            if self.generate_pairs:
                dataset2 = dataset.copy()
            # For each target, assign a random label.
            labels = np.random.randint(2, size=len(self.target_records)) == 1
            # If replace_target, then we first remove entries for the targets.
            if self.replace_target:
                # We first choose an entry e for each target record such that if
                # the target x is in the data, then e is removed (and vice versa).
                # This is to avoid replacing other target records. The reason we
                # do not use .replace_records is to be able to generate pairs.
                replace_indices = np.random.choice(
                    len(dataset), size=len(self.target_records), replace=False
                )
                # Remove the indices where label=1.
                dataset.drop_records(
                    [idx for idx, l in zip(replace_indices, labels) if l],
                    in_place=True,
                    n=0,  # Ensures that no records are dropped if the list is empty.
                )
                if self.generate_pairs:
                    # If generating pairs, remove indices where label=0 in dataset2.
                    dataset2.drop_records(
                        [idx for idx, l in zip(replace_indices, labels) if not l],
                        in_place=True,
                        n=0,  # Same as above.
                    )
            # Add the target records.
            for record, label in zip(self.target_records, labels):
                # If the label is 1, modify dataset.
                if label:
                    dataset.add_records(record, in_place=True)
                # If generating pairs and the label is 0, the label is 1 for
                # the other dataset in the pair. Modify dataset2
                elif self.generate_pairs:
                    dataset2.add_records(record, in_place=True)
            # Labels need to be converted, either as lists or int/float (if only one).
            _convert = list if len(self.target_records) > 1 else lambda x: x[0]
            mod_datasets.append(dataset)
            mod_labels.append(_convert(labels))
            if self.generate_pairs:
                mod_datasets.append(dataset2)
                mod_labels.append(_convert(labels == False))  # Negation.

        return mod_datasets, mod_labels

    @property
    def label(self):
        return self.attacker_knowledge.label


class TargetedMIA(LabelInferenceThreatModel):
    """
    This threat model implements a MIA with arbitrary attacker knowledge on
    data and generator.

    """

    def __init__(
        self,
        attacker_knowledge_data: AttackerKnowledgeOnData,
        target_record: Dataset,
        attacker_knowledge_generator: AttackerKnowledgeOnGenerator,
        generate_pairs: bool = True,
        replace_target: bool = False,
        memorise_datasets: bool = True,
        iterator_tracker: Option[type] = None,
        num_concurrent: int = 1,
    ):
        LabelInferenceThreatModel.__init__(
            self,
            MIALabeller(
                attacker_knowledge_data, target_record, generate_pairs, replace_target
            ),
            attacker_knowledge_generator,
            memorise_datasets,
            iterator_tracker=iterator_tracker,
            num_labels=len(target_record),
            num_concurrent=num_concurrent,
        )
        # Check that the targets are not already in the data (soft).
        self._assert_non_membership(target_record, attacker_knowledge_data)
        # Save the target recordS, and the current record (0).
        if self.multiple_label_mode:
            # Since calling .get_records creates a new Dataset object every
            # time, and involves indices, we instead compute the records once
            # and for all.
            self._target_records = [r for r in target_record]
            # This sets self.target_record.
            self.set_label(0)
        else:
            self.target_record = target_record

    def _assert_non_membership(self, target_record, attacker_knowledge_data):
        """
        Checks that target records are not used in the attacker knowledge's data.

        This does not raise an error but a warning that can be ignored. However,
        in most cases, it is recommended to ensure that target records are not also
        found in the auxiliary data, as this may make the task of inferring membership
        less meaningful: i.e., although "the" target record was not added to the
        dataset, another identical record is present in that data.

        """
        # Get all records used to simulate real training data.
        data_used = attacker_knowledge_data._get_data()
        num_records_found_in_data = sum([(r in data_used) for r in target_record])
        if num_records_found_in_data > 0:
            warnings.warn(
                f"{num_records_found_in_data} target record(s) were found in the auxiliary data. "
                + "This is not recommended: it is best to remove target records to avoid duplicates "
                + "and ensure that the task of membership inference is meaningful."
            )

    # Wrap the test method to output a MIAttackSummary.
    def _wrap_output(self, truth_labels, pred_labels, scores, attack):
        if self.num_labels > 1:
            target_id = ",".join([rec.label for rec in self._target_records])
        else:
            target_id = self.target_record.label
        return MIAttackSummary(
            truth_labels,
            pred_labels,
            scores,
            generator_info=self.atk_know_gen.label,
            attack_info=attack.label,
            dataset_info=self.atk_know_data.label,
            target_id=target_id,
        )

    def set_label(self, label: str):
        """
        If the attack is performed against multiple targets, this sets the
        target record to use when outputting labels.

        """
        # Use the parent class's set_label. The main reason we override this
        # method is to also modify self.target_record.
        LabelInferenceThreatModel.set_label(self, label)
        # We also set self.target_record, to be used by .
        self.target_record = self._target_records[label]

class PostHocThreatModelMIA(ThreatModel):
    """
    Some threat models considered using only synthetic datasets generated as 
    the attacker's knowledge.

    """
    def __init__(
        self,
        training_dataset: Dataset,
        synthetic_datasets: list[Dataset],
        reference_dataset: Dataset,
        membership_labels: list[int],   # 1 if target was in training set, 0 otherwise
        attacker_knowledge_data: AttackerKnowledgeOnData,
        target_record: Dataset,
        attacker_knowledge_generator: AttackerKnowledgeOnGenerator,
    ):
        
        self.synthetic_datasets = synthetic_datasets
        self.labels = membership_labels
        self.num_labels=len(target_record)
        self._target_records = [r for r in target_record]
        self.target_record = self._target_records[0]
        self.atk_know_gen = attacker_knowledge_generator
        self.atk_know_data = attacker_knowledge_data
        
    def set_label(self, i: int):
        self.target_record = self._target_records[i]

    def test(self, attack, *args, **kwargs):
        attack.threat_model = self
        attack.positive_label = 1
        attack.negative_label = 0

        all_truth = []
        all_pred = []
        all_scores = []

        for i in range(self.num_labels):
            self.set_label(i)
            pred_labels = attack.attack(self.synthetic_datasets)
            scores = attack.attack_score(self.synthetic_datasets)
            
            all_pred.append(pred_labels)
            all_scores.append(scores)

        # stack results per target
        truth = self.labels
        preds = np.vstack(all_pred)
        scores = np.vstack(all_scores)

        return self._wrap_output(truth, preds, scores, attack)

    def _wrap_output(self, truth_labels, pred_labels, scores, attack):
        if self.num_labels > 1:
            target_id = "all"#",".join([rec.label for rec in self._target_records])
        else:
            target_id = self.target_record.label
        return MIAttackSummary(
            truth_labels,
            pred_labels,
            scores,
            generator_info=self.atk_know_gen.label,
            attack_info=attack.label,
            dataset_info="Ground Truth",
            target_id=target_id,
        )
    
class NoBoxThreatModelMIA(ThreatModel):
    """
    Post Hoc Threat model for a no-box synthetic data scenario.
    
    Assumptions:
    - Attacker only has access to synthetic data (or multiple synthetic datasets).
    - Attacker may have auxiliary data from the same distribution.
    - Attacker wants to infer membership of records in the dataset used for training .
    """
    def __init__(self,
                attacker_knowledge_data: AttackerKnowledgeOnData, 
                attacker_knowledge_generator: AttackerKnowledgeOnGenerator,
                training_data: Dataset,
                num_targets: int = None,
                
    ):
        
        # Check that the targets are not already in the attackers knowledge data.
        self._assert_non_membership(training_data, attacker_knowledge_data)
        
        self.attacker_knowledge_data = attacker_knowledge_data
        self.atk_know_gen = attacker_knowledge_generator
        self.training_data = training_data
        self.num_targets = num_targets
        self.target_record = self._target_records[0]
        
        #self.member_frac = 0.5
        #self._target_records,self.true_labels = self._build_ground_truth()
        
        #self.num_labels = len(self._target_records) 

         
    def _build_ground_truth(
        self,
        member_indices: list = None
    ):
        """
        Construct target records and ground truth labels for evaluation.
        
        Returns
        -------
        target_records: Dataset
        true_labels: list[int]
        """
        
        # Uses a subset of auxiliary data (test data) as non members 
        test_data = self.attacker_knowledge_data.test_data
        
        if member_indices is None:
            num_members = min(len(self.training_data.data), len(test_data.data))
            members = self.training_data.sample(num_members)
            num_nonmembers = num_members # Maintain 50/50
        else:
            # Use the specific disjoint slice provided by the test loop
            members = self.training_data.get_records(member_indices)
            num_members = len(member_indices)
            # To maintain 50/50, we use an equal number of non-members
            # (Or use all non-members if you prefer, but 1:1 is standard)
            num_nonmembers = num_members
            
        # Ensure we don't request more non-members than we have
        num_nonmembers = min(num_nonmembers, len(test_data.data))
        nonmembers = test_data.sample(num_nonmembers)
        
        target_records = members.add_records(nonmembers)
        true_labels = [[1]] * len(members.data) + [[0]] * len(nonmembers.data)
                
        # if self.num_targets is None:
        #     count = min(len(self.training_data.data), len(test_data.data))
        #     num_members = int(count * self.member_frac)
        #     num_nonmembers = int(count * (1 - self.member_frac))
        
        # else:
        #     num_members = int(self.num_targets * self.member_frac)
        #     num_nonmembers = self.num_targets - num_members
        
        # if num_members > len(self.training_data.data):
        #     raise ValueError(f"Requested {num_members} members, but only {len(self.training_data.data)} exist.")
        # if num_nonmembers > len(test_data.data):
        #     raise ValueError(f"Requested {num_nonmembers} non-members, but only {len(test_data.data)} exist.")
        
        # # 2. Sample and combine
        # members = self.training_data.sample(num_members)
        # nonmembers = test_data.sample(num_nonmembers)
        
        # target_records = members.add_records(nonmembers)
        # true_labels = [[1]] * num_members + [[0]] * num_nonmembers
        
        # if num_targets is None:
        #     # Use all records
        #     target_records = training_data.add_records(test_data)
        #     true_labels = (
        #             [[1]] * len(training_data.data) +
        #             [[0]] * len(test_data.data)
        #         )
        # else:
        #     # Determine number of members and non-members
        #     num_nonmembers = min(int(num_targets * (1 - member_frac)), len(test_data.data))
        #     num_members = num_targets - num_nonmembers

        #     member_targets = training_data.sample(num_members)
        #     nonmember_targets = test_data.sample(num_nonmembers)

        #     target_records = member_targets.add_records(nonmember_targets)
        #     true_labels = [[1]] * num_members + [[0]] * num_nonmembers

        return target_records, true_labels


    def test(self, attack):

        """
        Evaluate an Attack object against this threat model using the test data.
        Returns metrics such as membership inference accuracy.
        """
               
        synthetic_datasets = self.atk_know_gen.generate(None, training_mode=False)
        
        # 2. Shuffle members once to ensure blocks are random but disjoint
        member_indices = list(range(len(self.training_data.data)))
        np.random.seed(42) 
        np.random.shuffle(member_indices)
        
        # Determine block size based on non-member availability to keep 1:1 ratio
        block_size = len(self.attacker_knowledge_data.test_data.data)
        summaries = []
        
        iterations = len(member_indices) // block_size
        
        for i in range(0,iterations):
            start_idx = i * block_size
            end_idx = start_idx + block_size
            
            current_block = member_indices[start_idx:end_idx]
            
            # Build ground truth for this specific 50/50 fold
            self._target_records, self.true_labels = self._build_ground_truth(member_indices=current_block)
            
            # 2. Get results for all targets at once
            # This calls SynthMiaTapasWrapper.attack_score()
            scores = attack.attack_score([synthetic_datasets])
            preds = attack.attack([synthetic_datasets])
            
            # 3. Align truth labels (Convert list of lists to numpy array)
            truth_labels = np.array(self.true_labels).reshape(-1, 1)
            
            summaries.append(self._wrap_output(truth_labels, preds, scores, attack, dataset_info='fold_{i}'))
        
        
        # for i in range(self.num_labels):
        #     self.set_label(i)
        #     pred_labels = attack.attack([synthetic_datasets])
        #     scores = attack.attack_score([synthetic_datasets])
            
        #     all_pred.append(pred_labels)
        #     all_scores.append(scores)

        # # stack results per target
        # truth_labels = np.vstack(self.true_labels)
        # preds = np.vstack(all_pred)
        # scores = np.vstack(all_scores)        

        return summaries

    def _wrap_output(self, truth_labels, pred_labels, scores, attack,dataset_info):
        target_id = "all"#",".join([rec.label for rec in self._target_records])
        return MIAttackSummary(
            truth_labels,
            pred_labels,
            scores,
            generator_info=self.atk_know_gen.label,
            attack_info=attack.label,
            dataset_info=dataset_info,
            target_id=target_id,
        )
        
    def set_label(self, i: int):
        self.target_record = self._target_records[i]

        