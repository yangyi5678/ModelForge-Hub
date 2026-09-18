from __future__ import annotations

from pathlib import Path
from typing import Any

from training_agent.data_preparation.services.local_paths import local_asset_path, write_jsonl
from training_agent.data_preparation.services.manifest_reader import iter_jsonl
from training_agent.data_preparation.services.object_store import ObjectStoreClient


def localize_assets(
    *,
    train_manifest_path: str,
    val_manifest_path: str,
    workspace: str,
    object_store: ObjectStoreClient,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Download manifest assets and emit localized manifests."""

    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    train_localized = _localize_split(
        manifest_path=train_manifest_path,
        workspace=workspace_path,
        split="train",
        object_store=object_store,
        skip_existing=skip_existing,
    )
    val_localized = _localize_split(
        manifest_path=val_manifest_path,
        workspace=workspace_path,
        split="val",
        object_store=object_store,
        skip_existing=skip_existing,
    )
    train_path = write_jsonl(workspace_path / "localized_train.jsonl", train_localized)
    val_path = write_jsonl(workspace_path / "localized_val.jsonl", val_localized)
    return {
        "train_dataset_uri": str(train_path),
        "validation_dataset_uri": str(val_path),
        "local_data_root": str(workspace_path / "files"),
        "model_dataset_format": "localized_manifest",
        "localized_train_manifest_path": str(train_path),
        "localized_val_manifest_path": str(val_path),
        "localized_train_count": len(train_localized),
        "localized_val_count": len(val_localized),
    }


def _localize_split(
    *,
    manifest_path: str,
    workspace: Path,
    split: str,
    object_store: ObjectStoreClient,
    skip_existing: bool,
) -> list[dict[str, Any]]:
    rows = []
    for index, row in enumerate(iter_jsonl(manifest_path), start=1):
        sample_id = str(row.get("sample_id") or f"{split}_{index:08d}")
        image_uri = _required_string(row, "image_oss_path")
        annotation_uri = _required_string(row, "annotation_oss_path")
        image_path = local_asset_path(
            workspace=workspace,
            split=split,
            asset_kind="images",
            source_uri=image_uri,
            sample_id=sample_id,
        )
        annotation_path = local_asset_path(
            workspace=workspace,
            split=split,
            asset_kind="annotations",
            source_uri=annotation_uri,
            sample_id=sample_id,
        )
        if not (skip_existing and image_path.exists()):
            object_store.download_file(image_uri, image_path)
        if not (skip_existing and annotation_path.exists()):
            object_store.download_file(annotation_uri, annotation_path)
        localized = dict(row)
        localized.update(
            {
                "local_image_path": str(image_path),
                "local_annotation_path": str(annotation_path),
            }
        )
        rows.append(localized)
    return rows


def _required_string(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest row missing required string field: {key}")
    return value
