"""
This example applies the DOMIAS attack to the setup of groundhog_census.py.
It shows how to integrate a custom attack into the TAPAS framework, train it,
evaluate it, and generate a report.
"""

import tapas.datasets
import tapas.generators
import tapas.threat_models
import tapas.report
import tapas.attacks

# Load the data.
data = tapas.datasets.TabularDataset.read(
    "data/adult/adult_test", label="Census"
)
target_record = data.get_records([1])
data.drop_records([1], in_place=True)

# Create a dummy generator (Raw just returns the same data it’s given).
generator = tapas.generators.Raw()

# Select the auxiliary data and synthetic data generation knowledge.
data_knowledge = tapas.threat_models.AuxiliaryDataKnowledge(
    data, auxiliary_split=0.5, num_training_records=1000
)

sdg_knowledge = tapas.threat_models.BlackBoxKnowledge(
    generator, num_synthetic_records=1000
)

threat_model = tapas.threat_models.TargetedMIA(
    attacker_knowledge_data=data_knowledge,
    target_record=target_record,
    attacker_knowledge_generator=sdg_knowledge,
    generate_pairs=True,
    replace_target=True,
)


# Instantiate DOMIASAttack.
attack = tapas.attacks.DOMIASAttack(density_estimator="bnaf")

# Train DOMIAS on the reference set from the threat model.
print(f"Training {attack.label}...")
attack.train(threat_model, num_samples=100)

# Evaluate the attack.
print("Evaluating DOMIAS...")
summary = threat_model.test(attack, num_samples=100)

# Publish MIA report.
print("Publishing a report.")
report = tapas.report.MIAttackReport([summary])
report.publish("domias_example")

# Publish ROC curve.
report = tapas.report.ROCReport([summary])
report.publish("domias_example")
