"""Training utilities for BABAPPA."""

from babappa.training.neural_data import (
    BabappaTensorDataset,
    NeuralDatasetConfig,
    SATURATION_TIER_TO_ID,
    collate_babappa_batch,
    inspect_neural_dataset,
    load_neural_rows,
    load_tensor_and_label,
    make_smoke_batch,
    resolve_tensor_file,
    saturation_tier_to_id,
)
from babappa.training.neural_env import (
    format_torch_environment_text,
    get_torch_environment,
    is_mps_available,
    resolve_torch_device,
    safe_import_torch,
)
from babappa.training.mps import (
    AppleSiliconBenchmarkConfig,
    MPSTrainingSmokeConfig,
    run_apple_silicon_benchmark,
    run_mps_training_smoke,
    validate_mps_smoke_dir,
)
from babappa.training.neural_audit import validate_neural_smoke_dir
from babappa.training.neural_full_audit import validate_neural_model_dir
from babappa.training.neural_model import (
    ContrastiveGeneClassifier,
    SaturationAwareGeneClassifier,
    SiteAttentionGeneClassifier,
    SmallGeneClassifier,
    build_gene_classifier,
    build_small_gene_classifier,
    count_parameters,
)
from babappa.training.losses import (
    bce_logits_loss,
    combined_loss,
    focal_bce_logits_loss,
    pairwise_rank_loss,
)
from babappa.training.neural_train import (
    NeuralTrainConfig,
    predict_neural_dataset,
    resolve_device,
    set_random_seeds,
    train_neural_smoke_model,
)
from babappa.training.neural_train_full import (
    NeuralFullTrainConfig,
    apply_training_preset,
    predict_neural_split,
    train_neural_model,
)

__all__ = [
    "BabappaTensorDataset",
    "AppleSiliconBenchmarkConfig",
    "NeuralDatasetConfig",
    "NeuralFullTrainConfig",
    "NeuralTrainConfig",
    "MPSTrainingSmokeConfig",
    "ContrastiveGeneClassifier",
    "SATURATION_TIER_TO_ID",
    "SaturationAwareGeneClassifier",
    "SiteAttentionGeneClassifier",
    "SmallGeneClassifier",
    "apply_training_preset",
    "build_gene_classifier",
    "build_small_gene_classifier",
    "bce_logits_loss",
    "collate_babappa_batch",
    "combined_loss",
    "count_parameters",
    "format_torch_environment_text",
    "focal_bce_logits_loss",
    "get_torch_environment",
    "inspect_neural_dataset",
    "is_mps_available",
    "load_neural_rows",
    "load_tensor_and_label",
    "make_smoke_batch",
    "pairwise_rank_loss",
    "predict_neural_split",
    "predict_neural_dataset",
    "resolve_device",
    "resolve_torch_device",
    "resolve_tensor_file",
    "run_apple_silicon_benchmark",
    "run_mps_training_smoke",
    "safe_import_torch",
    "saturation_tier_to_id",
    "set_random_seeds",
    "train_neural_model",
    "train_neural_smoke_model",
    "validate_neural_model_dir",
    "validate_neural_smoke_dir",
    "validate_mps_smoke_dir",
]
