import math
from typing import Any, Optional, Dict, cast
from torch.optim import Optimizer


def adjust_learning_rate(
        optimizer: Optimizer,
        epoch: int,
        args: Any,
) -> float:
    """Adjusts learning rate with half-cycle cosine decay after warmup.

    Args:
        optimizer: Optimizer whose learning rate will be updated.
        epoch: Current epoch number (0-indexed).
        args: Configuration object containing:
            - lr: Base learning rate
            - min_lr: Minimum learning rate
            - warmup_epochs: Number of warmup epochs
            - epochs: Total training epochs

    Returns:
        Updated learning rate (before applying lr_scale).
    """
    # Warmup phase: linear increase
    if epoch < args.warmup_epochs:
        lr = args.lr * (epoch / args.warmup_epochs)

    # Cosine decay phase: half-cycle cosine decay to min_lr
    else:
        decay_epochs = args.epochs - args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / decay_epochs
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = args.min_lr + (args.lr - args.min_lr) * cosine_factor

    # Apply learning rate with optional per-parameter scaling
    for param_group in optimizer.param_groups:
        param_group_cast = cast(Dict[str, Any], param_group)
        lr_scale = param_group_cast.get("lr_scale", 1.0)
        param_group_cast["lr"] = lr * lr_scale

    return lr