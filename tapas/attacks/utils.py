from . import GroundhogAttack, ClosestDistanceMIA, ProbabilityEstimationAttack, LpDistance
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity


def evaluate_basic_available_attacks(threat_model, additional_attacks=[]):
    """Evaluates a set of predefined basic attacks on the given threat model."""
    attacks = [
        GroundhogAttack(
            use_hist=False, use_corr=False, label="NaiveGroundhog"
        ),
        GroundhogAttack(
            use_naive=False, use_corr=False, label="HistGroundhog"
        ),
        GroundhogAttack(
            use_naive=False, use_hist=False, label="CorrGroundhog"
        ),
        GroundhogAttack(
            model=LogisticRegression(), label="LogisticGroundhog"
        ),
        ClosestDistanceMIA(
            criterion="accuracy", label="ClosestDistance-Hamming"
        ),
        ClosestDistanceMIA(
            criterion=("threshold", 0), label="Direct Lookup"
        ),
        ClosestDistanceMIA(
            distance=LpDistance(2), criterion="accuracy", label="ClosestDistance-L2"
        ),
        ProbabilityEstimationAttack(
            KernelDensity(), criterion="accuracy", label="KernelEstimator"
        ),   
    ] + additional_attacks

    
    # Train, evaluate, and summarise all attacks.
    summaries = []
    for attack in attacks:
        print(f"Evaluating attack {attack.label}...")
        try:
            attack.train(threat_model, num_samples=100)
            summaries.append(threat_model.test(attack, num_samples=100))
        except Exception as e:
            print(f"\tEvaluating attack {attack.label} failed with '{e}', excluding it from the report ...")
            
    return summaries