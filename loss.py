from torch.nn import CrossEntropyLoss

from config import Config


def get_loss_function(config: Config, pad_id: int):
    loss_type = config.loss.type.lower()

    if loss_type == "cross_entropy":
        label_smoothing = config.loss.label_smoothing
        return CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=pad_id)
    raise NotImplementedError(f"Unsupported loss type: {loss_type}")
