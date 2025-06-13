import itertools


import tapas.datasets
import tapas.generators
import tapas.threat_models
import tapas.attacks
import tapas.report

from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_absolute_percentage_error, r2_score

CAT_METRCIS = {
    'acc': accuracy_score,
    'f1': f1_score
}

NUM_METRICS = {
    'mae': mean_absolute_error,
    'r2': r2_score,
    'mape': mean_absolute_percentage_error
}


def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")

real_dataset = tapas.datasets.TabularDataset.read(
    "data/adult/adult_data", label="Adult"
)
synth_dataset = tapas.datasets.TabularDataset.read(
    "data/adult/adult_synthetic_data", label="Adult"
)

generator = tapas.generators.NoBoxGenerator(real_dataset, 'RealDataGenerator')
generator2 = tapas.generators.NoBoxGenerator(synth_dataset, 'SynthetiDataGenerator')

target_record = real_dataset.get_records([1])
real_dataset.drop_records([1], in_place=True)
# Select the auxiliary data + black-box attack model.
data_knowledge = tapas.threat_models.ExactDataKnowledge(
    real_dataset
)

sdg_knowledge = tapas.threat_models.BlackBoxKnowledge(
    generator, num_synthetic_records=1,
)
sdg_knowledge2 = tapas.threat_models.BlackBoxKnowledge(
    generator2, num_synthetic_records=1,
)

threat_model = tapas.threat_models.TargetedAIA(
    attacker_knowledge_data=data_knowledge,
    # Specific to AIA: the sensitive attribute and its possible values.
    sensitive_attribute="sex",
    attribute_values=[0, 1],
    target_record=target_record,
    attacker_knowledge_generator=sdg_knowledge,
)
threat_model_extended = tapas.threat_models.extend_threat_model(threat_model, [accuracy_score, f1_macro])

threat_model2 = tapas.threat_models.TargetedAIA(
    attacker_knowledge_data=data_knowledge,
    # Specific to AIA: the sensitive attribute and its possible values.
    sensitive_attribute="sex",
    attribute_values=[0, 1],
    target_record=target_record,
    attacker_knowledge_generator=sdg_knowledge2,
)
threat_model_extended2 = tapas.threat_models.extend_threat_model(threat_model2, [accuracy_score, f1_macro])

quasi_ids = ['education','workclass','occupation']
attacks = [
    tapas.attacks.MLAttack(quasi_ids, 'Dummy'),
    tapas.attacks.MLAttack(quasi_ids, 'RF'),
    tapas.attacks.MLAttack(quasi_ids, 'SVM'),
    tapas.attacks.MLAttack(quasi_ids, 'NB'),
    tapas.attacks.MLAttack(quasi_ids, 'KNN'),
    tapas.attacks.MLAttack(quasi_ids, 'LR'),
    tapas.attacks.MLAttack(quasi_ids)
]

summaries = []
for attack in attacks:
    print(f"Evaluating attack {attack.label}...")
    attack.train(threat_model_extended)
    summaries.append(threat_model_extended.test(attack, num_samples=100))
    attack.train(threat_model_extended2)
    summaries.append(threat_model_extended2.test(attack, num_samples=100))

report = tapas.report.MIAttackReport(summaries, metrics=['accuracy', 'f1_macro', 'auc'])
print(report.attacks_data)

report.publish('attribute_disclosure')



    
    
    
    