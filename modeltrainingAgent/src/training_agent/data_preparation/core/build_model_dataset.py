from __future__ import annotations


def build_model_dataset(
    *,
    train_dataset_uri: str,
    validation_dataset_uri: str,
    model_family: str,
) -> dict[str, str]:
    """Build model-specific dataset files.

    For now the model dataset is a manifest passthrough. Concrete converters
    such as Qwen-VL can later replace these URIs with generated training JSONL.
    """

    return {
        "train_dataset_uri": train_dataset_uri,
        "validation_dataset_uri": validation_dataset_uri,
        "model_family": model_family,
    }

