from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from vlm_project.data import VLMDataset
from vlm_project.data_collector import VLMDataCollator
from vlm_project.entrypoints import build_loss, build_model, build_optimizer, build_scheduler


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "qwen25_vl.yaml"


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("config must be a YAML mapping")
    return data


def resolve_device(cfg: dict[str, Any]) -> torch.device:
    configured = cfg.get("training", {}).get("device")
    if configured:
        return torch.device(configured)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def build_dataloader(cfg: dict[str, Any], processor) -> DataLoader:
    dataset = VLMDataset(cfg, processor)
    collator = VLMDataCollator(processor)
    train_cfg = cfg["training"]
    return DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=train_cfg.get("num_workers", 0),
    )


def save_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scheduler,
    cfg: dict[str, Any],
    epoch: int,
    global_step: int,
    loss: float | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": cfg,
        "loss": loss,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(
    path: str | Path, *, model, optimizer, scheduler, device: torch.device
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint["epoch"]) + 1, int(checkpoint.get("global_step", 0))


def evaluate_loss(model, loss_fn, dataloader: DataLoader, device: torch.device, max_steps: int | None):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if max_steps is not None and step >= max_steps:
                break
            batch = move_to_device(batch, device)
            loss = loss_fn(model, batch)
            total_loss += float(loss.detach().cpu())
            count += 1
    model.train()
    return total_loss / max(count, 1)


def train(cfg: dict[str, Any], resume_checkpoint: str | None = None) -> Path:
    device = resolve_device(cfg)
    model, processor = build_model(cfg)
    model.to(device)

    dataloader = build_dataloader(cfg, processor)
    loss_fn = build_loss(cfg)
    optimizer = build_optimizer(model, cfg)

    train_cfg = cfg["training"]
    epochs = int(train_cfg["epochs"])
    grad_accum = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
    updates_per_epoch = max(1, math.ceil(len(dataloader) / grad_accum))
    total_steps = updates_per_epoch * epochs
    scheduler = build_scheduler(optimizer, cfg, total_steps)

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    global_step = 0
    if resume_checkpoint:
        start_epoch, global_step = load_checkpoint(
            resume_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

    max_steps_per_epoch = train_cfg.get("max_steps_per_epoch")
    model.train()

    for epoch in range(start_epoch, epochs):
        optimizer.zero_grad(set_to_none=True)
        last_loss: float | None = None

        for step, batch in enumerate(dataloader):
            if max_steps_per_epoch is not None and step >= int(max_steps_per_epoch):
                break

            batch = move_to_device(batch, device)
            loss = loss_fn(model, batch)
            scaled_loss = loss / grad_accum
            scaled_loss.backward()
            last_loss = float(loss.detach().cpu())

            is_accum_boundary = (step + 1) % grad_accum == 0
            is_last_batch = step + 1 == len(dataloader)
            if is_accum_boundary or is_last_batch:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    train_cfg["max_grad_norm"],
                )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            print(f"epoch={epoch} step={step} loss={last_loss:.4f}")

        save_checkpoint(
            output_dir / "last.ckpt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            epoch=epoch,
            global_step=global_step,
            loss=last_loss,
        )
        if train_cfg.get("eval_each_epoch", False):
            eval_loss = evaluate_loss(
                model,
                loss_fn,
                dataloader,
                device,
                train_cfg.get("max_eval_steps"),
            )
            print(f"epoch={epoch} eval_loss={eval_loss:.4f}")

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--resume-checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, resume_checkpoint=args.resume_checkpoint)


if __name__ == "__main__":
    main()
