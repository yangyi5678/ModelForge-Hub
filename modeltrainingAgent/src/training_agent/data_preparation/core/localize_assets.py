from __future__ import annotations

from pathlib import Path


def localize_assets(
    *,
    train_manifest_path: str,
    val_manifest_path: str,
    workspace: str,
    mode: str = "manifest_passthrough",
) -> dict[str, str]:
    """Prepare local assets.

    The first production-safe version keeps manifests as the dataset contract.
    Image/annotation downloading and shard extraction can be added behind the
    same return contract without changing downstream nodes.
    """

    Path(workspace).mkdir(parents=True, exist_ok=True)
    if mode != "manifest_passthrough":
        raise ValueError(f"unsupported asset localization mode: {mode}")
    return {
        "train_dataset_uri": train_manifest_path,
        "validation_dataset_uri": val_manifest_path,
        "local_data_root": str(Path(workspace) / "files"),
        "model_dataset_format": mode,
    }

