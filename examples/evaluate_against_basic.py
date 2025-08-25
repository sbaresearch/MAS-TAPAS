"""
This example applies several attacks to tthe setup of groundhog_census.py, and
shows how to write code without duplicating unnecessary elements, and how to
handle reports with multiple attacks.

"""

import tapas.datasets
import tapas.generators
import tapas.threat_models
import tapas.report
import tapas.attacks


# Load the data.
data = tapas.datasets.TabularDataset.read(
    "data/2011 Census Microdata Teaching File", label="Census"
)
target_record = data.get_records([1])
data.drop_records([1], in_place=True)


# Create a dummy generator.
generator = tapas.generators.Raw()

# Select the auxiliary data + black-box attack model.
data_knowledge = tapas.threat_models.AuxiliaryDataKnowledge(
    data, auxiliary_split=0.5, num_training_records=1000,
)

sdg_knowledge = tapas.threat_models.BlackBoxKnowledge(
    generator, num_synthetic_records=1000,
)

# Create the threat model.
threat_model = tapas.threat_models.TargetedAIA(
    attacker_knowledge_data=data_knowledge,
    sensitive_attribute="sex",
    attribute_values=["1", "2"],
    target_record=target_record,
    attacker_knowledge_generator=sdg_knowledge,
)

# Train, evaluate, and summarise basic attacks.
summaries = tapas.attacks.evaluate_basic_available_attacks(threat_model)

# Finally, group together the summaries as a report.
print("Publishing a report.")
report = tapas.report.MIAttackReport(summaries)
print(report.attacks_data)
report.publish("multiple_mia")
