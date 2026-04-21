"""Reusable metric library: distances, auth metrics, profiles, stability."""

from .basic import (
    coefficient_of_variation,
    correlation_coefficient,
    euclidean_distance,
    normalized_cross_correlation,
    pairwise_euclidean,
    pairwise_ncc,
    shannon_entropy,
    transmission_loss_db,
)
from .group import (
    inter_class_distance,
    intra_class_distance,
    intra_inter_ratio,
    within_class_similarity,
)
from .auth import (
    auc_score,
    confusion_matrix,
    equal_error_rate,
    nearest_neighbor_identify,
    roc_curve,
    top_k_accuracy,
)
from .profile import (
    fit_gaussian_profile,
    profile_width,
    radial_intensity_profile,
)
from .stability import (
    aggregate_mean_std,
    bootstrap_ci,
    temporal_stability_score,
)

__all__ = [
    "coefficient_of_variation",
    "correlation_coefficient",
    "euclidean_distance",
    "normalized_cross_correlation",
    "pairwise_euclidean",
    "pairwise_ncc",
    "shannon_entropy",
    "transmission_loss_db",
    "inter_class_distance",
    "intra_class_distance",
    "intra_inter_ratio",
    "within_class_similarity",
    "auc_score",
    "confusion_matrix",
    "equal_error_rate",
    "nearest_neighbor_identify",
    "roc_curve",
    "top_k_accuracy",
    "fit_gaussian_profile",
    "profile_width",
    "radial_intensity_profile",
    "aggregate_mean_std",
    "bootstrap_ci",
    "temporal_stability_score",
]
