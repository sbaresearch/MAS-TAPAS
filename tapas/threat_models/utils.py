from ..report import ExtendedAttackSummary, MIAttackSummary, AIAttackSummary, BinaryAIAttackSummary
from .aia import TargetedAIA

def extend_threat_model(threat_model, extra_metrics):
    
    cls = type(threat_model)

    ReportClass = MIAttackSummary
    kwargs = {}
    if isinstance(threat_model, TargetedAIA):
        kwargs = {"sensitive_attribute": threat_model.sensitive_attribute}
        if len(threat_model.attribute_values) == 2:
            ReportClass = BinaryAIAttackSummary
            kwargs["positive_value"] = threat_model.attribute_values[1]
        else:
            ReportClass = AIAttackSummary

    class ExtendedClass(cls):
        def __init__(self, super_obj, extra_metrics):
            self.extra_metrics = extra_metrics
            for k, v in vars(super_obj).items():
                setattr(self, k, v)

        def _wrap_output(self, truth_labels, pred_labels, scores, attack):
            if len(self.extra_metrics) == 0:
                return cls._wrap_output(self, truth_labels, pred_labels, scores, attack)
            
            return ExtendedAttackSummary(
                ReportClass,
                truth_labels,
                pred_labels,
                scores,
                generator_info=self.atk_know_gen.label,
                attack_info=attack.label,
                dataset_info=self.atk_know_data.label,
                target_id=self.target_record.label,
                extra_metrics = self.extra_metrics,
                **kwargs
            )
        
    return ExtendedClass(threat_model, extra_metrics)