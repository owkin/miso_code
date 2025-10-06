from typing import List, Optional

import torch
from miso.models.layers import TilesMLP
from torch import nn


class SimpleMIL(nn.Module):
    """Wrapper around the implementation of classic_algos.

    Parameters
    ----------
    input_dim: int
        Input dimension
    output_dim: int
        Output dimension, must match the number of genes to predict.
    hidden: Optional[List[int]] = None
        List of the layers' dimensions, by default None.
    dropout: Optional[List[float]] = None
        List of the dropout rates, by default None.
    activation: Optional[torch.nn.Module] = torch.nn.Sigmoid()
        Activation function, by default torch.nn.Sigmoid().
    bias: bool = True
        Whether to use bias, by default True.
    bias_init: Optional[torch.Tensor] = None
        Tensor initiating the bias MLP's last layer with the mean expression of the
        training dataset, by default None.
    agg_method: str = "mean"
        Whether to use mean or max pooling, by default "mean".
    """

    def __init__(  # pylint: disable=unused-argument
        self,
        input_dim: int,
        output_dim: int,
        hidden: Optional[List[int]] = None,
        dropout: Optional[List[float]] = None,
        activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        bias: bool = True,
        bias_init: Optional[torch.Tensor] = None,
        agg_method: str = "mean",
        device: str = "cuda",
        **kwargs
    ):
        super().__init__()
        self.mlp = TilesMLP(
            in_features=input_dim,
            out_features=output_dim,
            hidden=hidden,
            dropout=dropout,
            activation=activation,
            bias=bias,
        )
        self.agg_method = agg_method
        if bias_init is not None:
            self.mlp[-1].bias = bias_init

        self.device = device
        self.to(self.device)

    def _mean(self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None):
        if mask is not None:
            # Only mean over non padded features
            mean_x = torch.sum(x.masked_fill(mask, 0.0), dim=1) / torch.sum(
                (~mask).float(), dim=1
            )
        else:
            mean_x = torch.mean(x, dim=1)
        return mean_x
        # return torch.mean(x, dim=1)

    def _max(self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None):
        if mask is not None:
            # Only max over non padded features
            max_x, _ = torch.max(x.masked_fill(mask, float("-inf")), dim=1)
        else:
            max_x, _ = torch.max(x, dim=1)
        return max_x

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        """Forward function.

        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, IN_FEATURES)
        mask: Optional[torch.BoolTensor]
            (B, N_TILES, 1), True for values that were padded.

        Returns
        -------
        logits: torch.Tensor
            (B, OUT_FEATURES)
        """
        x = self.mlp(x)
        x = torch.nn.ReLU()(x)

        if self.agg_method == "max":
            return self._max(x, mask)
        elif self.agg_method == "mean":
            return self._mean(x, mask)
        raise ValueError(f"Aggregation method {self.agg_method} not supported.")
