import torch
from torch import nn


class ResidualAddNorm(nn.Module):
    """
    Applies residual addition followed by layer normalization.

    This module is designed to perform a residual connection operation by
    adding an input tensor (residual) to a sublayer output tensor. The
    resultant tensor is then normalized using layer normalization.

    The normalization is implemented using PyTorch's `nn.LayerNorm`, which
    provides an option to use learned affine parameters (scale and bias). The
    epsilon parameter ensures numerical stability during normalization.

    :ivar norm_layer: Layer normalization instance that applies normalization to
        the combined residual and sublayer output.
    :type norm_layer: nn.LayerNorm
    """

    def __init__(self, n_features: int, eps: float = 1e-5, learn_affine: bool = True):
        """
        Initializes the normalization layer with specified parameters.

        The class creates a LayerNorm instance from the PyTorch nn module. It provides
        flexibility in choosing the number of features, epsilon value for numerical
        stability, and whether affine parameters (scale and bias) are learned.

        :param n_features: Number of input features to normalize.
        :type n_features: Int
        :param eps: Small constant added to denominator for numerical stability
            during normalization.
        :type eps: Float, optionally
        :param learn_affine: Specifies whether to learn affine parameters.
        :type learn_affine: Bool, optional
        """
        super().__init__()
        self.norm_layer = nn.LayerNorm(n_features, eps, elementwise_affine=learn_affine)

    def forward(
        self, residual: torch.Tensor, sublayer_output: torch.Tensor
    ) -> torch.Tensor:
        add = residual + sublayer_output
        return self.norm_layer(add)
