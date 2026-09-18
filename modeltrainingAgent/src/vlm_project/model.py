import torch

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

from peft import (
    LoraConfig,
    get_peft_model,
)


def build_model(cfg):
    model_cfg = cfg["model"]

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    dtype = dtype_map[model_cfg.get("dtype", "bfloat16")]

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_cfg["path"],
        torch_dtype=dtype,
    )

    processor = AutoProcessor.from_pretrained(
        model_cfg["path"]
    )

    mode = model_cfg["finetune_mode"]

    # -------------------------
    # Full SFT
    # -------------------------
    if mode == "full":

        for param in model.parameters():
            param.requires_grad = True

    # -------------------------
    # Partial SFT
    # -------------------------
    elif mode == "partial":

        for param in model.parameters():
            param.requires_grad = False

        train_modules = model_cfg["partial"]["train_modules"]

        for name, param in model.named_parameters():
            if any(module in name for module in train_modules):
                param.requires_grad = True

    # -------------------------
    # LoRA SFT
    # -------------------------
    elif mode == "lora":

        lora_cfg = model_cfg["lora"]

        peft_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(
            model,
            peft_config,
        )

    else:
        raise ValueError(
            f"Unknown finetune_mode: {mode}"
        )

    if cfg["training"].get(
        "gradient_checkpointing",
        False,
    ):
        model.gradient_checkpointing_enable()

    return model, processor