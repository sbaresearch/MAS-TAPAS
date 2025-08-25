import mlflow
from mlflow import pyfunc
import cloudpickle
import pandas as pd
from sys import version_info
import tapas.datasets
import tapas.generators
import tapas.threat_models
import tapas.attacks
import tapas.report

"""
Before runnig the experiment, one should start MLflow tracking server with the following command:
    `mlflow server --host 127.0.0.1 --port 5000`
Host and port can be different, but then `TRACKING_URL` should be adapted accordingly.
One could add `--default-artifact-root ./mlruns` to the command to prevent multiple artifacts download.
"""

TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "SimpleSDG"

# Define simple SDG model
class SimpleSDG(pyfunc.PythonModel):
    def load_context(self, context):
        if getattr(self, "_initialized", False):
            print("Dataset already loaded")
            return
        
        with open(context.artifacts["dataset"], "rb") as f:
            self.dataset: pd.DataFrame = cloudpickle.load(f)

    def predict(self, context, model_input, params=None):
        n = int(model_input) if isinstance(model_input, int) else 100

        synthetic = pd.DataFrame()
        for col in self.dataset.columns:
            synthetic[col] = self.dataset[col].sample(n=n, replace=True).reset_index(drop=True)
        return synthetic


def save_dataset(csv_path, output_path="artifacts/tabular_dataset.pkl"):
    df = pd.read_csv(csv_path)

    with open(output_path, "wb") as f:
        cloudpickle.dump(df, f)

    return output_path

def register_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("synthetic-data-experiment")

    conda_env = {
        "channels": ["defaults"],
        "dependencies": [
            f"python={version_info.major}.{version_info.minor}.{version_info.micro}",
            "pip",
            {
                "pip": [
                    f"mlflow=={mlflow.__version__}",
                    f"cloudpickle=={cloudpickle.__version__}",
                    "pandas"
                ],
            },
        ],
        "name": "sdg_env",
    }

    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path=MODEL_NAME,
            python_model=SimpleSDG(),
            artifacts={"dataset": save_dataset("data/adult/adult_data.csv")},
            conda_env=conda_env,
            registered_model_name=MODEL_NAME
        )

register_model()


data = tapas.datasets.TabularDataset.read(
    "data/adult/adult_test", label="Adult Test"
)
target_record = data.get_records([1])
data.drop_records([1], in_place=True)

# Create a generator.
generator = tapas.generators.GeneratorFromMLflow(
    tracking_uri=TRACKING_URI,
    model_name=MODEL_NAME
)

# Select the auxiliary data + black-box attack model.
data_knowledge = tapas.threat_models.AuxiliaryDataKnowledge(
    data, auxiliary_split=0.5, num_training_records=1000,
)

sdg_knowledge = tapas.threat_models.BlackBoxKnowledge(
    generator, num_synthetic_records=1000,
)

threat_model = tapas.threat_models.TargetedAIA(
    attacker_knowledge_data=data_knowledge,
    # Specific to AIA: the sensitive attribute and its possible values.
    sensitive_attribute="Sex",
    attribute_values=["1", "2"],
    target_record=target_record,
    attacker_knowledge_generator=sdg_knowledge,
)

# We here create a range of attacks to test.
attacks = [
    tapas.attacks.GroundhogAttack(
        use_hist=False, use_corr=False, label="NaiveGroundhog"
    ),
    tapas.attacks.GroundhogAttack(
        use_naive=False, use_corr=False, label="HistGroundhog"
    ),
    tapas.attacks.GroundhogAttack(
        use_naive=False, use_hist=False, label="CorrGroundhog"
    ),
    tapas.attacks.ClosestDistanceMIA(
        criterion="accuracy", label="ClosestDistance-Hamming"
    ),
    tapas.attacks.ClosestDistanceMIA(
        criterion=("threshold", 0), label="Direct Lookup"
    ),
]

# Train, evaluate, and summarise all attacks.
summaries = []
for attack in attacks:
    print(f"Evaluating attack {attack.label}...")
    attack.train(threat_model, num_samples=100)
    summaries.append(threat_model.test(attack, num_samples=100))

# Finally, group together the summaries as a report.
print("Publishing a report.")
report = tapas.report.MIAttackReport(summaries)
report.publish("generatorFromMLflow_aia")