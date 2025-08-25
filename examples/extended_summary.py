import tapas.datasets
import tapas.generators
import tapas.threat_models
import tapas.attacks
import tapas.report
from sklearn.metrics import precision_score, recall_score, f1_score

# extra metrics 
def f1_micro(y_true, y_pred):
    return f1_score(y_true, y_pred, average="micro")

def recall_weighted(y_true, y_pred):
    return recall_score(y_true, y_pred, average="weighted")

# Load the data.
data = tapas.datasets.TabularDataset.read(
    "data/2011 Census Microdata Teaching File", label="Census"
)

# Selects the target record
target_record = data.get_records([1])
data.drop_records([1], in_place=True)

# Create a dummy generator (It just samples data from original dataset)
generator = tapas.generators.Raw()

# Select the auxiliary data + black-box attack model.
data_knowledge = tapas.threat_models.AuxiliaryDataKnowledge(
    data, auxiliary_split=0.5, num_training_records=1000,
)

sdg_knowledge = tapas.threat_models.BlackBoxKnowledge(
    generator, num_synthetic_records=1000,
)

threat_model_extended = tapas.threat_models.TargetedAIA(
    attacker_knowledge_data=data_knowledge,
    # Specific to AIA: the sensitive attribute and its possible values.
    sensitive_attribute="Country of Birth",
    attribute_values=["-9", "1", "2"],
    target_record=target_record,
    attacker_knowledge_generator=sdg_knowledge
)
threat_model_extended = tapas.threat_models.extend_threat_model(threat_model_extended, [f1_micro, recall_weighted])

attack_extended =tapas.attacks.GroundhogAttack()
attack_extended.train(threat_model_extended, num_samples=100)

summary_extended = threat_model_extended.test(attack_extended, num_samples=100)
report_extended = tapas.report.MIAttackReport([summary_extended], metrics=['accuracy', 'f1_micro', 'recall_weighted'])  
print(report_extended.attacks_data)
report_extended.publish("extended_aia")
