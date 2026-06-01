import logging

import torch
from torch.utils.data import DataLoader

from model import TransformerModel
from utils import BatchTensors

logger = logging.getLogger(__name__)

# def train_with_BLEU(model: Model, dataset, batch_size: int, epoch: int):
#     pass


def train_epoch_avg_CE(
    model: TransformerModel,
    train_data_loader: DataLoader[BatchTensors],
    loss_fn,
    wandb_run=None,
    max_batches: int = 100,
) -> torch.Tensor:
    """
    Computes the average cross-entropy loss per batch for a training epoch.

    This function iterates through the training data loader, processes each batch by
    moving data to the device associated with the model, and calculates per-batch loss
    using the provided loss function. The computed batch losses are aggregated and
    averaged over the total number of batches to determine the mean loss for the epoch.
    Memory optimized version that limits the number of batches processed to prevent
    RAM overflow.

    Args:
        model (TransformerModel): The model to train. Must have a `device` attribute
            where computations should occur.
        train_data_loader (DataLoader[BatchTensors]): DataLoader object yielding batches
            of training data. Each batch provides tensors for source input, target input,
            target labels, and padding masks.
        loss_fn: A callable that computes the loss given model predictions and target
            values. Should return a tensor containing the computed loss value.
        wandb_run: The wandb run object to log to. If None, no logging will be done.
        max_batches (int): Maximum number of batches to process to prevent memory
            overflow. Defaults to 100.

    Returns:
        torch.Tensor: A tensor representing the average loss per batch for the training
            epoch. If no batches exist in the loader, returns `float('inf')` as the loss.

    Note:
        The function includes memory management features such as CUDA cache clearing
        and OOM error handling to prevent system crashes during training.
    """
    batch_loss = 0.0
    num_batches = 0

    for idx, batch in enumerate(train_data_loader, start=1):
        # Limit number of batches to prevent memory overflow
        if idx >= max_batches:
            logger.info(f"Reached maximum batch limit of {max_batches}, stopping epoch")
            break

        batch_on_device = BatchTensors(
            src_batch_X=batch.src_batch_X.to(model.device),
            tgt_batch_X=batch.tgt_batch_X.to(model.device),
            tgt_batch_y=batch.tgt_batch_y.to(model.device),
            src_batch_X_pad_mask=batch.src_batch_X_pad_mask.to(model.device),
            tgt_batch_X_pad_mask=batch.tgt_batch_X_pad_mask.to(model.device),
        )

        try:
            loss = train_batch_CE(
                model=model, batch=batch_on_device, loss_fn=loss_fn
            ).item()
            logger.info(f"Batch: {idx} loss: {loss: .5f}")

            # Wandb logging
            if wandb_run is not None:
                wandb_run.log({"batch_loss": loss, "batch_idx": idx})

            batch_loss += loss
            num_batches += 1

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(
                    f"GPU out of memory at batch {idx}. Try reducing batch_size or model size."
                )
                # Clear cache and continue
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break
            raise e
        finally:
            # Clear cache after each batch to prevent memory accumulation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if num_batches == 0:
        return torch.tensor(float("inf"), device=model.device, dtype=torch.float32)
    # Return average loss per batch
    return torch.tensor(
        batch_loss / num_batches, device=model.device, dtype=torch.float32
    )


def train_batch_CE(
    model: TransformerModel, batch: BatchTensors, loss_fn
) -> torch.Tensor:
    """
    Performs a single training step through the transformer model with cross-entropy loss.

    This function processes one batch of data through the complete transformer pipeline,
    including encoder, decoder, and classifier components. It computes the cross-entropy
    loss for the batch and returns it for further processing (like gradient computation).

    Args:
        model (TransformerModel): The transformer model containing encoder, decoder,
            and classifier components.
        batch (BatchTensors): Named tuple containing:
            - src_batch_X: Source sequences tensor [batch_size, seq_len]
            - tgt_batch_X: Target input sequences tensor [batch_size, seq_len]
            - tgt_batch_y: Target output sequences tensor [batch_size, seq_len]
            - src_batch_X_pad_mask: Source padding mask [batch_size, seq_len]
            - tgt_batch_X_pad_mask: Target padding mask [batch_size, seq_len]
        loss_fn: Loss function that computes cross-entropy loss. Expected to take
            logits and targets as arguments.

    Returns:
        torch.Tensor: The computed loss value for the batch. Returns infinite loss
            if no loss function is provided.

    Note:
        The function transposes logits from [B, S, V] to [B, V, S] format as required
        by PyTorch's CrossEntropyLoss function.
    """
    src_batch_X = batch.src_batch_X
    tgt_batch_X = batch.tgt_batch_X
    tgt_batch_y = batch.tgt_batch_y
    src_batch_X_pad_mask = batch.src_batch_X_pad_mask
    tgt_batch_X_pad_mask = batch.tgt_batch_X_pad_mask

    logits = model.forward(
        src_batch_X, tgt_batch_X, src_batch_X_pad_mask, tgt_batch_X_pad_mask
    )  # [BATCH_SIZE, SEQ_LEN, VOCAB_SIZE]

    # Calculate loss if a loss function is provided
    # CrossEntropy Loss expects the logits for all the inputs in the batch of size [BATCH_SIZE, VOCAB_SIZE, SEQ_LEN]
    # So we transpose the logits
    if loss_fn is not None:
        return loss_fn(logits.transpose(1, 2), tgt_batch_y)  # [B, S, V] -> [B, V, S]

    # Return infinite loss
    return torch.tensor(float("inf"), device=logits.device)
