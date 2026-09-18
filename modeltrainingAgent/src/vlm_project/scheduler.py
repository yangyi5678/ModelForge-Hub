from transformers import get_scheduler


def build_scheduler(
    optimizer,
    cfg,
    num_training_steps,
):

    scheduler_cfg = cfg["scheduler"]

    warmup_ratio = scheduler_cfg.get(
        "warmup_ratio",
        0.03,
    )

    num_warmup_steps = int(
        num_training_steps * warmup_ratio
    )

    scheduler = get_scheduler(
        name=scheduler_cfg.get(
            "type",
            "cosine",
        ),
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    return scheduler