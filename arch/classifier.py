import torch
from torch import nn

from config import Config


class Classifier(nn.Module):
    """
    Neural network classifier for vocabulary prediction in transformer models.

    This class implements a simple linear classifier that maps hidden representations
    to vocabulary logits. It consists of a single fully connected layer that transforms
    the input features into logits corresponding to each token in the vocabulary.
    The class is designed for use with cross-entropy loss, so no softmax activation
    is applied in the forward pass.

    Attributes:
        linear (nn.Linear): Fully connected layer that maps input features to
            output logits with dimensions [hidden_size, vocab_size].
    """

    def __init__(self, hidden_size: int, vocab_size: int):
        """
        Initialize the classifier with specified dimensions.

        Args:
            hidden_size (int): Size of the input feature dimension.
            vocab_size (int): Size of the output vocabulary.
        """
        super().__init__()
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classifier.

        Applies a linear transformation to convert hidden representations into
        vocabulary logits. No softmax is applied as this is typically handled
        by the loss function (e.g., CrossEntropyLoss).

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size].

        Returns:
            torch.Tensor: Output logits of shape [batch_size, seq_len, vocab_size].

        Note:
            Softmax is not applied here as it's typically included in the loss
            function for numerical stability when using CrossEntropyLoss.
        """
        logits = self.linear(x)  # [BATCH, SEQ_LEN, VOCAB_SIZE]
        # We don't need to apply softmax because we are using CrossEntropyLoss
        # prob = self.softmax(logits) # [BATCH, SEQ_LEN, VOCAB_SIZE]
        return logits


def get_classifier(conf: Config) -> Classifier:
    """
    Create a classifier instance based on configuration parameters.

    This factory function extracts the necessary parameters from the configuration
    object and initializes a Classifier instance with the appropriate dimensions.

    Args:
        conf (Config): Configuration object containing model parameters. Must have
            accessible attributes:
            - conf.model.hidden_size: Hidden dimension size
            - conf.model.vocab_size: Vocabulary size

    Returns:
        Classifier: Initialized classifier instance ready for training or inference.
    """
    hidden_size: int = conf.model.hidden_size
    vocab_size: int = conf.model.vocab_size
    return Classifier(hidden_size, vocab_size)
