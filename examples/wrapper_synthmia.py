"""
This example demonstrates how to run membership inference attacks (MIA)
from the [Synth-MIA](https://github.com/joshward96/Synth-MIA) library through the TAPAS wrapper, supported only in the NoBox setting.
"""
from sklearn.metrics import precision_score
import tapas.datasets
import tapas.attacks
import tapas.threat_models


# Load the datasets for the attacks.
real_dataset = tapas.datasets.TabularDataset.read("data/adult/adult_data", label="Adult").sample(1000)
synth_dataset = tapas.datasets.TabularDataset.read("data/adult/adult_synthetic_data", label="Synthetic Data Adult").sample(1000)
holdout_dataset = tapas.datasets.TabularDataset.read("data/adult/adult_test", label="Auxiliary Data Adult").sample(500)

# Define auxiliary data with the holdout dataset
data_knowledge = tapas.threat_models.AuxiliaryDataKnowledge(
    dataset=holdout_dataset,          # auxiliary dataset from same distribution
    auxiliary_split=0.5,          # attacker knows half of the underlying distribution   
)

# Deine the generator as NoBox only access to synthetic dataset (not generator)
generator = tapas.generators.NoBoxGenerator(synth_dataset)
sdg_knowledge = tapas.threat_models.NoBoxKnowledge(
    generator=generator,  
    num_synthetic_records=None
)

# List the SYNTHMIA attacks using the SynthMiaTapasWrapper  
params_attack_1={'k_nearest':5}
att1 = tapas.attacks.SynthMiaTapasWrapper(attacker_name= 'GenLRA' ,**params_attack_1)
att2 = tapas.attacks.SynthMiaTapasWrapper(attacker_name= 'DCR' )             
att3 = tapas.attacks.SynthMiaTapasWrapper(attacker_name= 'LOGAN')
att4 = tapas.attacks.SynthMiaTapasWrapper(attacker_name='DCRDiff')
att5 = tapas.attacks.SynthMiaTapasWrapper(attacker_name='DOMIAS')
att6 = tapas.attacks.SynthMiaTapasWrapper(attacker_name='MC')
params_attack_7={'method':'kde'}
att7 = tapas.attacks.SynthMiaTapasWrapper(attacker_name='DensityEstimate',**params_attack_7)
att8 = tapas.attacks.SynthMiaTapasWrapper(attacker_name='LocalNeighborhood')
att9 = tapas.attacks.SynthMiaTapasWrapper(attacker_name='Classifier')
att10 = tapas.attacks.SynthMiaTapasWrapper(attacker_name='DPI')
attacks = [att1, att2, att3, att4, att5, att6, att7, att8, att9, att10]

# Instantiate the NoBoxThreatModel MIA
threat_model = tapas.threat_models.NoBoxThreatModelMIA(
    attacker_knowledge_data= data_knowledge,
    attacker_knowledge_generator= sdg_knowledge, 
    target_records=real_dataset, # All the training records are targeted in the MIA attacks
    target_data= 'all' 
)

# Evaluate the attacks against the threat model extended with additional metrics       
def precision_macro(y_true, y_pred):
 return precision_score(y_true, y_pred, average="macro", zero_division=0)
threat_model_extended = tapas.threat_models.extend_threat_model(threat_model, [precision_macro])
summaries = []
for attack in attacks: 
    try:
        attack.train(threat_model_extended)
        summaries.append(threat_model_extended.test(attack))
    except Exception as e:
        print(f'{e} error str{attack}')
        
print("Publishing a report.")
report = tapas.report.MIAttackReport(summaries)
print(report.attacks_data)
report.publish("synthmia_results")




