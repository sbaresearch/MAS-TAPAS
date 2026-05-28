"""A test for threat models."""

import asyncio
import os
from unittest import TestCase

import itertools
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from tapas.datasets import TabularDataset, TabularRecord
from tapas.datasets.data_description import DataDescription
from tapas.generators.generator import NoBoxGenerator
from tapas.report.attack_summary import AIAttackSummary, BinaryAIAttackSummary, MIAttackSummary
from tapas.threat_models import (
    ThreatModel,
    TargetedMIA,
    TargetedAIA,
    AuxiliaryDataKnowledge,
    BlackBoxKnowledge,
    NoBoxKnowledge,
    UncertainBoxKnowledge,
)
from tapas.generators import Raw, Generator
from tapas.threat_models.aia import NoBoxThreatModelAIA
from tapas.threat_models.attacker_knowledge import AttackerKnowledgeOnData
from tapas.threat_models.mia import NoBoxThreatModelMIA


class RawConcurrent(Raw):
    """A generator that's like Raw, but uses a coroutine to return results async."""

    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    @property
    def label(self):
        return "RawConcurrent"


dummy_data_description = DataDescription(
    [
        {"name": "a", "type": "countable", "description": "integer"},
        {"name": "b", "type": "countable", "description": "integer"},
        {"name": "c", "type": "countable", "description": "integer"},
    ]
)

dummy_data = pd.DataFrame(
    [(0, 1, 0), (0, 2, 1), (3, 4, 0), (3, 5, 1), (6, 6, 1)], columns=["a", "b", "c"]
)

dataset = TabularDataset(dummy_data, dummy_data_description)

# Choose the target record (4), and remove it from the dataset.
target_record = dataset.get_records([4])
dataset = dataset.drop_records([4])

knowledge_on_data = AuxiliaryDataKnowledge(
    dataset, auxiliary_split=0.5, num_training_records=2
)
knowledge_on_sdg = BlackBoxKnowledge(Raw(), num_synthetic_records=None)
knowledge_on_sdg_concurrent = BlackBoxKnowledge(
    RawConcurrent(), num_synthetic_records=None
)


class TestMIA(TestCase):
    """Test the membership-inference attack."""

    def _test_labelling_helper(self, generate_pairs, replace_target, async_generator, use_concurrency):
        """Test whether the datasets are correctly labelled."""
        if not async_generator and use_concurrency:
            # This combination of parameters doesn't make sense.
            return None
        atk_know = knowledge_on_sdg_concurrent if async_generator else knowledge_on_sdg
        num_concurrent = 5 if use_concurrency else 1
        mia = TargetedMIA(
            knowledge_on_data,
            target_record,
            atk_know,
            generate_pairs=generate_pairs,
            replace_target=replace_target,
            num_concurrent=num_concurrent,
        )
        self.assertEqual(mia.multiple_label_mode, False)
        # Check that we generate the correct number of samples.
        num_samples = 100
        datasets, labels = mia.generate_training_samples(num_samples)
        self.assertEqual(len(datasets), num_samples)
        self.assertEqual(len(datasets), len(labels))
        # We here use RAW as a generator, so the datasets generated are the
        # training datasets directly. We can thus check target membership on
        # the dataset and that the labels are correct.
        for ds, target_in in zip(datasets, labels):
            self.assertEqual(len(ds), 2 if (replace_target or not target_in) else 3)
            self.assertEqual(target_record in ds, target_in)

    def test_labelling_default(self):
        self._test_labelling_helper(False, False, False, False)

    def test_labelling_pairs(self):
        self._test_labelling_helper(True, False, False, False)

    def test_labelling_replace(self):
        self._test_labelling_helper(False, True, False, False)

    def test_labelling_replace_pairs(self):
        self._test_labelling_helper(True, True, False, False)

    def test_labelling_default_async(self):
        self._test_labelling_helper(False, False, True, False)

    def test_labelling_pairs_async(self):
        self._test_labelling_helper(True, False, True, False)

    def test_labelling_replace_async(self):
        self._test_labelling_helper(False, True, True, False)

    def test_labelling_replace_pairs_async(self):
        self._test_labelling_helper(True, True, True, False)

    def test_labelling_default_threads(self):
        self._test_labelling_helper(False, False, False, True)

    def test_labelling_pairs_threads(self):
        self._test_labelling_helper(True, False, False, True)

    def test_labelling_replace_threads(self):
        self._test_labelling_helper(False, True, False, True)

    def test_labelling_replace_pairs_threads(self):
        self._test_labelling_helper(True, True, False, True)

    def test_labelling_default_async_threads(self):
        self._test_labelling_helper(False, False, True, True)

    def test_labelling_pairs_async_threads(self):
        self._test_labelling_helper(True, False, True, True)

    def test_labelling_replace_async_threads(self):
        self._test_labelling_helper(False, True, True, True)

    def test_labelling_replace_pairs_async_threads(self):
        self._test_labelling_helper(True, True, True, True)


class TestMIAMultipleTargets(TestCase):
    """Test the membership-inference attack with multiple targets."""

    def _test_multiple_targets(self, generate_pairs, replace_target):
        # Some parameters.
        num_training_records = 100
        num_targets = 10
        # Generate all combinations (so that records are unique!).
        large_dummy_data = TabularDataset(
            pd.DataFrame(
                list(itertools.product(range(10), range(10), range(10))),
                columns=["a", "b", "c"],
            ),
            dummy_data_description,
        )
        # Select a large number of targets.
        target_idxs = np.random.choice(
            len(large_dummy_data), size=(num_targets,), replace=False
        )
        target_records = large_dummy_data.get_records(target_idxs)
        large_dummy_data.drop_records(target_idxs, in_place=True)
        # Create the threat model with multiple targets.
        mia = TargetedMIA(
            AuxiliaryDataKnowledge(
                large_dummy_data,
                auxiliary_split=0.5,
                num_training_records=num_training_records,
            ),
            target_records,
            knowledge_on_sdg,
            replace_target=replace_target,
            generate_pairs=generate_pairs,
        )
        self.assertEqual(mia.multiple_label_mode, True)
        # Generate datasets and check the labelling.
        for r, threat_model_targeted in zip(target_records, mia):
            # Check that the target record is properly set.
            self.assertEqual(len(threat_model_targeted.target_record), 1)
            for x, y in zip(
                threat_model_targeted.target_record.data.values[0], r.data.values[0]
            ):
                self.assertEqual(x, y)
            # Generate some datasets (unchanged through raw).
            num_generated_samples = 20
            datasets, labels = threat_model_targeted.generate_training_samples(
                num_generated_samples
            )
            self.assertEqual(len(datasets), num_generated_samples)
            self.assertEqual(len(labels), num_generated_samples)
            # Check that the datasets are properly labelled.
            for ds, target_in in zip(datasets, labels):
                # If targets are replaced, the dataset should always have the
                # same numbers of records.
                if replace_target:
                    self.assertEqual(len(ds), num_training_records)
                elif target_in:
                    # If not replacing, and this record has been *added* to the
                    # dataset, the size of ds is larger than the number of records.
                    # Note that we can't know the expected size without having
                    # access to all labels.
                    self.assertGreater(len(ds), num_training_records)
                self.assertEqual(r in ds, target_in)

    def test_labelling_default(self):
        self._test_multiple_targets(False, False)

    def test_labelling_pairs(self):
        self._test_multiple_targets(True, False)

    def test_labelling_replace(self):
        self._test_multiple_targets(False, True)

    def test_labelling_replace_pairs(self):
        self._test_multiple_targets(True, True)


class TestAIA(TestCase):
    """Test the attribute-inference attack."""

    def test_labelling(self):
        """Test whether the datasets are correctly labelled."""
        aia = TargetedAIA(
            knowledge_on_data, target_record, "c", [0, 1], knowledge_on_sdg
        )
        num_samples = 100
        datasets, labels = aia.generate_training_samples(num_samples)
        self.assertEqual(len(datasets), num_samples)
        self.assertEqual(len(datasets), len(labels))
        for ds, target_value in zip(datasets, labels):
            record = target_record.copy()
            record.set_value("c", target_value)
            self.assertEqual(record in ds, True)

    def test_multiple_targets(self):
        """Test whether the datasets are correctly labelled, for multiple targets."""
        num_training_records = 100
        num_targets = 10
        # Generate all combinations (so that records are unique!), but with many more
        # values for (a,b) --> 900 records, and "c" being binary.
        large_dummy_data = TabularDataset(
            pd.DataFrame(
                list(itertools.product(range(30), range(30), (0, 1))),
                columns=["a", "b", "c"],
            ),
            dummy_data_description,
        )
        # Select a large number of targets.
        target_idxs = np.random.choice(
            len(large_dummy_data), size=(num_targets,), replace=False
        )
        target_records = large_dummy_data.get_records(target_idxs)
        large_dummy_data.drop_records(target_idxs, in_place=True)
        # Create the threat model.
        aia = TargetedAIA(
            AuxiliaryDataKnowledge(
                large_dummy_data,
                auxiliary_split=0.5,
                num_training_records=num_training_records,
            ),
            target_records,
            "c",
            [0, 1],
            knowledge_on_sdg,
        )
        # Generate datasets and check the labelling.
        for r, threat_model_targeted in zip(target_records, aia):
            # Check that the target record is found in the dataset.
            self.assertEqual(len(threat_model_targeted.target_record), 1)
            for x, y, col in zip(
                threat_model_targeted.target_record.data.values[0],
                r.data.values[0],
                ["a", "b", "c"],
            ):
                # Check equality for non-sensitive attributes.
                if col != "c":
                    self.assertEqual(x, y)
            # Generate some datasets (unchanged through raw).
            num_generated_samples = 20
            datasets, labels = threat_model_targeted.generate_training_samples(
                num_generated_samples
            )
            self.assertEqual(len(datasets), num_generated_samples)
            self.assertEqual(len(labels), num_generated_samples)
            # Check that the record with the correct value is found in the dataset.
            for ds, value in zip(datasets, labels):
                record = r.copy()
                record.set_value("c", value)
                self.assertEqual(record in ds, True)


class TestAttackerKnowledge(TestCase):
    """Test the attacker knowledge."""

    def test_auxiliary_dataset(self):
        gen_data = lambda size: TabularDataset(
            pd.DataFrame(
                np.random.randint(10, size=(size, 3)), columns=["a", "b", "c"]
            ),
            dummy_data_description,
        )
        # Check that the auxiliary and test datasets have appropriate size.
        for aux_size, test_size, split, full_size in [
            (20, 20, 0.5, 1000),
            (0, 0, 0.1, 100),
            (117, 39, 0.8, None),
        ]:
            dataset = gen_data(full_size) if full_size is not None else None
            threat_model = AuxiliaryDataKnowledge(
                dataset=dataset,
                auxiliary_split=split,
                aux_data=gen_data(aux_size) if aux_size > 0 else None,
                test_data=gen_data(test_size) if test_size > 0 else None,
            )
            # Compute the contribution of the full dataset to auxiliary and test data.
            aux_split_size = int(split * full_size) if full_size is not None else 0
            test_split_size = full_size - aux_split_size if full_size is not None else 0
            # Check that sizes are as expected.
            self.assertEqual(len(threat_model.aux_data), aux_size + aux_split_size)
            self.assertEqual(len(threat_model.test_data), test_size + test_split_size)

    def test_no_box(self):
        gen = NoBoxKnowledge(Raw(), 2)
        with pytest.raises(Exception) as err:
            gen(dataset, training_mode=True)
        gen(dataset, training_mode=False)

    def test_uncertain_box(self):
        # First, define a silly 1-dimensional generator.
        class Replicator(Generator):
            def __call__(self, dataset, num_samples, mean=0):
                return np.full((num_samples,), mean)
            def fit(self, *args): pass
            def generate(self, *args): pass

        # Then, define a threat model using this, and test it.
        gen = UncertainBoxKnowledge(
            Replicator(), 1, lambda: {"mean": np.random.normal()}, {"mean": 117}
        )
        records_train = [gen(None, training_mode = True) for _ in range(1000)]
        records_test = [gen(None, training_mode = False) for _ in range(1000)]
        self.assertTrue(np.mean(records_train) < 4)  # Unlikely to fail.
        self.assertTrue(np.std(records_train) < 2)
        for x in records_test:
            self.assertEqual(x[0], 117)


class TestMemory(TestCase):
    """Check whether saving/loading threat models works, as well as saving in memory."""

    def test_save_then_load(self):
        output_dir = os.path.join(os.path.dirname(__file__), "outputs")
        name = os.path.join(output_dir, "threat_model_test")
        os.makedirs(output_dir, exist_ok=True)
        
        threat_model = TargetedMIA(
            knowledge_on_data,
            target_record,
            knowledge_on_sdg,
            generate_pairs=False,
            replace_target=False,
        )
        training_samples = 103
        testing_samples = 42
        threat_model.generate_training_samples(training_samples)
        threat_model._generate_samples(testing_samples, False)
        threat_model.save(name)
        threat_model_2 = ThreatModel.load(name)
        # Check that the models are identical:
        self.assertIsInstance(threat_model_2, TargetedMIA)
        # Check that the target records are the same.
        for x, y in zip(
            threat_model.target_record.data, threat_model_2.target_record.data
        ):
            self.assertEqual(x, y)
        # The following is specific to TargetedMIA, and checks that the internal
        # memory of the object is properly set. This is the most important
        # feature of .save and .load: to not have to recompute the datasets.
        self.assertEqual(len(threat_model_2._memory[True][0]), training_samples)
        self.assertEqual(len(threat_model_2._memory[False][0]), testing_samples)
        # Check that the samples are identical (from memory:).
        datasets1, labels1 = threat_model._memory[True]
        datasets2, labels2 = threat_model_2._memory[True]
        for l1, l2 in zip(labels1, labels2):
            self.assertEqual(l1, l2)
        for d1, d2 in zip(datasets1, datasets2):
            self.assertEqual(len(d1), len(d2))
            for x, y in zip(d1, d2):
                for v1, v2 in zip(x.data.values[0], y.data.values[0]):
                    self.assertEqual(v1, v2)
        # Finally, check that the name is properly set.
        self.assertEqual(threat_model_2._name, name)


    def test_memory(self):
        threat_model = TargetedMIA(
            knowledge_on_data,
            target_record,
            knowledge_on_sdg,
            generate_pairs=False,
            replace_target=False,
        )
        # Without using memory.
        for training in [True, False]:
            for num_samples in [4, 7, 11, 20]:
                mem_datasets, mem_labels = threat_model._generate_samples(
                    num_samples, training, ignore_memory=True
                )
                self.assertEqual(len(mem_datasets), num_samples)
                self.assertEqual(len(mem_labels), num_samples)
                # Check that the memory is empty.
                self.assertEqual(len(threat_model._memory[training][0]), 0)
                self.assertEqual(len(threat_model._memory[training][1]), 0)
        # Using memory.
        for training in [True, False]:
            for num_samples in [4, 7, 11, 20]:
                mem_datasets, mem_labels = threat_model._generate_samples(
                    num_samples, training, ignore_memory=False
                )
                self.assertEqual(len(mem_datasets), num_samples)
                self.assertEqual(len(mem_labels), num_samples)
                # Check that the memory has the right size.
                self.assertEqual(len(threat_model._memory[training][0]), num_samples)
                self.assertEqual(len(threat_model._memory[training][1]), num_samples)

class TestNoBoxThreatModelAIA(TestCase):
    """Test the attribute-inference no box threat model."""
    
    def setUp(self):
        self.data_desc = DataDescription([
            {"name": "age", "type": "countable", "description": "integer"},
            {"name": "zip", "type": "countable", "description": "integer"},
            {"name": "salary", "type": "countable", "description": "sensitive"},
        ])

        # Setup Target Records (3 records)
        target_df = pd.DataFrame([
            (25, 1001, 50),
            (30, 1002, 60),
            (35, 1003, 70)
        ], columns=["age", "zip", "salary"])
        self.target_dataset = TabularDataset(target_df, self.data_desc)

        # Setup Attacker Knowledge on Data (with test/control records)
        control_df = pd.DataFrame([
            (40, 1004, 80),
            (45, 1005, 90)
        ], columns=["age", "zip", "salary"])
        control_ds = TabularDataset(control_df, self.data_desc)
        
        self.atk_know_data = MagicMock(spec=AuxiliaryDataKnowledge)
        self.atk_know_data.test_data = control_ds

        mock_gen = MagicMock(spec=NoBoxGenerator)
        self.atk_know_gen = NoBoxKnowledge(mock_gen,None)

        # 5. Initialize Threat Model
        self.threat_model = NoBoxThreatModelAIA(
            target_records=self.target_dataset,
            sensitive_attribute="salary",
            quasi_identifiers=["age", "zip"],
            attacker_knowledge_data=self.atk_know_data,
            attacker_knowledge_generator=self.atk_know_gen,
            attribute_values=[50, 60, 70, 80, 90]
        )
    
    def test_initialization(self):
        """Test if attributes are correctly assigned and filtered to relevant columns."""
        self.assertEqual(len(self.threat_model._target_records), 3)
        self.assertEqual(len(self.threat_model._control_records), 2)
        # Check if view filtering worked (relevant_cols = zip, age, salary)
        first_record_cols = self.threat_model.target_record.data.columns.tolist()
        self.assertIn("salary", first_record_cols)
        self.assertIn("age", first_record_cols)
        self.assertEqual(len(first_record_cols), 3)

    def test_set_label(self):
        """Test switching between target and control records."""
        # Test target group
        self.threat_model.set_label(1, group='target')
        self.assertEqual(self.threat_model.target_record.data["age"].iloc[0], 30)
        
        # Test control group
        self.threat_model.set_label(0, group='control')
        self.assertEqual(self.threat_model.target_record.data["age"].iloc[0], 40)

    def test_nobox_assertions(self):
        """Test that the model strictly enforces NoBox types."""
        wrong_gen_knowledge = BlackBoxKnowledge(Raw(), num_synthetic_records=None)
        
        with self.assertRaises(AssertionError):
            NoBoxThreatModelAIA(
                self.target_dataset, "salary", ["age"], 
                self.atk_know_data, wrong_gen_knowledge
            )

    def test_attack_flow(self):
        """Test the execution of the attack loop and output wrapping."""
        mock_attack = MagicMock()
        mock_attack.label = "MockAttack"

        mock_attack.attack.return_value = [60] 
        mock_attack.attack_score.return_value = [0.8]

        synthetic_ds = TabularDataset(
            pd.DataFrame([(30, 1002, 60)], columns=["age", "zip", "salary"]),
            self.data_desc
        )
        self.atk_know_gen.generate = MagicMock(return_value=synthetic_ds)

        report = self.threat_model.test(mock_attack)
        self.assertEqual(mock_attack.attack.call_count, 5) # Number of times called (target records + control records)
        self.assertEqual(len(report.predictions), 3) # Based on target_records
        

    def test_wrap_output_binary_logic(self):
        """Test if the correct Summary class is chosen for binary vs multiclass."""
        
        mock_attack = MagicMock()
        mock_attack.label = "MockAttack"
        
        # Binary case
        binary_model = NoBoxThreatModelAIA(
            target_records=self.target_dataset,
            sensitive_attribute="salary",
            quasi_identifiers=["age"],
            attacker_knowledge_data=self.atk_know_data,
            attacker_knowledge_generator=self.atk_know_gen,
            attribute_values=[0, 1] # Binary
        )
        
        binary_report = binary_model._wrap_output(
            truth_labels=[0, 1], 
            pred_labels=[1, 0], 
            scores=[0.9,0.4], 
            attack=mock_attack
        )
        self.assertIsInstance(binary_report, BinaryAIAttackSummary)
        
        multiclass_model = NoBoxThreatModelAIA(
            target_records=self.target_dataset,
            sensitive_attribute="salary",
            quasi_identifiers=["age"],
            attacker_knowledge_data=self.atk_know_data,
            attacker_knowledge_generator=self.atk_know_gen,
            attribute_values=[0, 1, 2]  # Three values
        )
        
        multi_report = multiclass_model._wrap_output(
            truth_labels=[0, 2], 
            pred_labels=[0, 1], 
            scores=[[0.6,0.3,0.1], [0.1,0.8,0.1] ], # Multiclass scores usually structured differently
            attack=mock_attack
        )
        self.assertIsInstance(multi_report, AIAttackSummary)
        
class TestNoBoxThreatModelMIA(TestCase):
    """Test the membership-inference no box threat model."""

    def setUp(self):
        self.data_desc = DataDescription([
            {"name": "age", "type": "countable", "description": "integer"},
            {"name": "zip", "type": "countable", "description": "integer"},
            {"name": "salary", "type": "countable", "description": "integer"},
        ])

        # Setup Training Records (3 records - mapped as 'members')
        training_df = pd.DataFrame([
            (25, 1001, 50),
            (30, 1002, 60),
            (35, 1003, 70)
        ], columns=["age", "zip", "salary"])
        self.training_dataset = TabularDataset(training_df, self.data_desc)
        # Ensure underlying mock datasets can support mock sampling method
        self.training_dataset.sample = MagicMock(return_value=self.training_dataset)

        # Setup Control Records inside Attacker Knowledge (2 records - mapped as 'non-members')
        control_df = pd.DataFrame([
            (40, 1004, 80),
            (45, 1005, 90)
        ], columns=["age", "zip", "salary"])
        self.control_dataset = TabularDataset(control_df, self.data_desc)
        self.control_dataset.sample = MagicMock(return_value=self.control_dataset)
        
        # Mock Attacker Knowledge on Data
        self.atk_know_data = MagicMock(spec=AuxiliaryDataKnowledge)
        self.atk_know_data.test_data = self.control_dataset
        self.atk_know_data._get_data.return_value = [] # Avoid warning triggers during init

        # Mock Attacker Knowledge on Generator
        mock_gen = MagicMock(spec=NoBoxGenerator)
        self.atk_know_gen = NoBoxKnowledge(mock_gen, None)
        

        # Initialize Threat Model under test
        self.threat_model = NoBoxThreatModelMIA(
            attacker_knowledge_data=self.atk_know_data,
            attacker_knowledge_generator=self.atk_know_gen,
            target_records=self.training_dataset,
            target_data='all'
        )

    def test_initialization(self):
        """Test if basic configuration properties are correctly assigned on init."""
        self.assertEqual(self.threat_model.num_labels, 3)
        self.assertEqual(self.threat_model.target_data, 'all')
        self.assertEqual(self.threat_model.training_data, self.training_dataset)


    def test_attack_flow(self):
        """Test execution of threat evaluation, tracking model calling signatures."""
        mock_attack = MagicMock()
        mock_attack.label = "MockMIAttack"
        mock_attack.attack.return_value = [1, 0, 1, 0, 0]
        mock_attack.attack_score.return_value = [0.9, 0.1, 0.85, 0.3, 0.2]

        synthetic_ds = MagicMock(spec=TabularDataset)
        self.atk_know_gen.generate = MagicMock(return_value=synthetic_ds)

        # Mock the internal builder to isolate testing to pipeline flow execution
        mock_eval_dataset = MagicMock(spec=TabularDataset)
        mock_labels = [[1], [1], [1], [0], [0]]
        self.threat_model._build_ground_truth = MagicMock(return_value=(mock_eval_dataset, mock_labels))

        report = self.threat_model.test(mock_attack, balance=False)

        # Assert correct components passed downwards
        self.atk_know_gen.generate.assert_called_once_with(None, training_mode=False)
        mock_attack.attack.assert_called_once_with([synthetic_ds])
        mock_attack.attack_score.assert_called_once_with([synthetic_ds])

        # Assert wrapping object returns correctly typed report instance
        self.assertIsInstance(report, MIAttackSummary)
