
import torch

from uniperceiver.utils.registry import Registry

EVALUATION_REGISTRY = Registry("EVALUATION")
EVALUATION_REGISTRY.__doc__ = """
Registry for evaluation
"""

def build_evaluation(cfg, annfile, output_dir):
    evaluation = EVALUATION_REGISTRY.get(cfg.INFERENCE.NAME)(cfg, annfile, output_dir) if len(cfg.INFERENCE.NAME) > 0 else None
    return evaluation