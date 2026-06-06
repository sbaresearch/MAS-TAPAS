"""
This example is similar to multiple_attacks.py, except that it applies
attribute inference attacks (AIA) applicable only with the NoBoxThreatModelAIA.
"""

from sklearn.metrics import accuracy_score

import tapas.datasets
import tapas.generators
import tapas.threat_models
import tapas.attacks
import tapas.report
import numpy as np
import itertools

# Load the datasets for the attacks.
real_dataset = tapas.datasets.TabularDataset.read("data/adult/adult_data", label="Adult").sample(1000)
synth_dataset = tapas.datasets.TabularDataset.read("data/adult/adult_synthetic_data", label="Synthetic Data Adult").sample(1000)
holdout_dataset = tapas.datasets.TabularDataset.read("data/adult/adult_test", label="Auxiliary Data Adult").sample(500)

# Define auxiliary data with the holdout dataset
data_knowledge = tapas.threat_models.AuxiliaryDataKnowledge(
    dataset=holdout_dataset,          # auxiliary dataset from same distribution
    auxiliary_split=0.5,            
)

# Define generator as NoBox
generator = tapas.generators.NoBoxGenerator(synth_dataset)    
sdg_knowledge = tapas.threat_models.NoBoxKnowledge(generator,1000)


# Define quasi-identifiers and sensitive attributes
quasi_identifiers=['age','education','race']
sensitive_attribute='income'

# Consider multiple combinations of quasi-identifiers known by an attacker to model multiple attack scenarios
all_combinations = []
for r in range(1, len(quasi_identifiers) + 1):
    # Generate all combinations of length r
    combinations_object = itertools.combinations(quasi_identifiers, r)
    combinations_list = list(combinations_object)
    all_combinations.extend(combinations_list)
    
# Define threat models for all combinations
all_threat_models = []
for combination in all_combinations:
    threat_model = tapas.threat_models.NoBoxThreatModelAIA(
        target_records=real_dataset,
        sensitive_attribute=sensitive_attribute,
        attribute_values=['<=50K','>50K'],
        quasi_identifiers=list(combination),
        attacker_knowledge_generator=sdg_knowledge,
        attacker_knowledge_data=data_knowledge,
        target_data='all'
    )
    all_threat_models.append(threat_model)



# Add additional metrics to AIA metrics (Optional) 
def accuracy_baseline(y_true, y_pred):
    values, counts = np.unique(y_true, return_counts=True)
    ind = np.argmax(counts)
    most_frequent_val = values[ind]
    y_baseline_pred = np.full(y_true.shape, fill_value=most_frequent_val)
    return accuracy_score(y_true, y_baseline_pred)

def attacker_advantage(y_true,y_pred):
    acc_base=accuracy_baseline(y_true,y_pred)
    acc_real=accuracy_score(y_true,y_pred)
    return acc_real-acc_base

def risk_score(y_true,y_pred,context):
    # Extract the extra data from kwargs
    control_labels =context['control_labels']
    control_preds = context['control_preds']
    
    acc_target = accuracy_score(y_true,y_pred)
    acc_control = accuracy_score(control_labels,control_preds)
    if acc_control >= 1.0:
        return 0.0
    risk = (acc_target - acc_control) / (1.0 - acc_control)
    return risk 

def accuracy_control(y_true,y_pred, context):
    # Extract the extra data from kwargs
    control_labels =context['control_labels']
    control_preds = context['control_preds']
    
    control_labels=context.get("control_labels")
    control_preds=context.get("control_preds")
    acc_control = accuracy_score(control_labels,control_preds)
    return acc_control 

#Extend threat models with additional metrics 
all_threat_models_extended = []
for threat_model_ in all_threat_models:
    threat_model_extended = tapas.threat_models.extend_threat_model(threat_model_, [accuracy_baseline,attacker_advantage,risk_score,accuracy_control])
    all_threat_models_extended.append(threat_model_extended)

# Train, evaluate, and summarise all attacks for the quasi-identifiers combinations (including extended metrics).
summaries=[]
for threat_model_extended in all_threat_models_extended:
    # Define all the list of attacks available for AIA 
    attack1 = tapas.attacks.ClosestDistanceAIA(criterion=("threshold", -0.5))
    attack2 = tapas.attacks.LocalNeighbourhoodAttack(radius=3,criterion=("threshold", 0.6))
    attack3 = tapas.attacks.GeneralizedCAPAttack()
    attack4 = tapas.attacks.MLInferenceAttack()
    aia_attacks = [attack1,attack2,attack3,attack4]
    for attack in aia_attacks: 
        try:
            attack.train(threat_model_extended)
            summaries.append(threat_model_extended.test(attack))
            print(attack)
        except Exception as e:
            print(f'{e} error str{attack}')
            
print("Publishing a report.")
report = tapas.report.AIAttackReport(summaries)
print(report.attacks_data)
report.publish("aia_results")