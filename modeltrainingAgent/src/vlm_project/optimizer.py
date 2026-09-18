import torch


def build_optimizer(model, cfg):

    train_cfg = cfg["training"]

    trainable_params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    return optimizer