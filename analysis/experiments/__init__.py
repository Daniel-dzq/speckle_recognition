"""All paper-section experiments exposed from a single namespace."""

from .base import BaseExperiment, ExperimentContext
from ._features import CaptureFeature, extract_features
from .authentication import AuthenticationExperiment
from .common_mode import CommonModeExperiment
from .demo import DemoExperiment
from .dual_channel import DualChannelExperiment
from .length_optimization import LengthOptimizationExperiment
from .system_setup import SystemSetupExperiment

EXPERIMENT_REGISTRY = {
    "system_setup": SystemSetupExperiment,
    "length_optimization": LengthOptimizationExperiment,
    "dual_channel": DualChannelExperiment,
    "common_mode": CommonModeExperiment,
    "authentication": AuthenticationExperiment,
    "demo": DemoExperiment,
}

__all__ = [
    "BaseExperiment",
    "ExperimentContext",
    "CaptureFeature",
    "extract_features",
    "AuthenticationExperiment",
    "CommonModeExperiment",
    "DemoExperiment",
    "DualChannelExperiment",
    "LengthOptimizationExperiment",
    "SystemSetupExperiment",
    "EXPERIMENT_REGISTRY",
]
