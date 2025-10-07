"""
Copyright (c) Owkin Inc.
This source code is licensed under the CC BY-NC-SA 4.0 license found in the
LICENSE file in the root directory of this source tree.

Implementation of local attention MIL from  https://arxiv.org/abs/2205.06672.
"""
import torch

from typing import Optional, List

from miso.models.layers import LocalSelfAttention, MaskedLinear, TilesMLP


class LocalAttentionMIL(torch.nn.Module):
    """Local attention MIL (https://arxiv.org/abs/2205.06672).

    Example:
        >>> module = LocalAttentionMIL(in_features=128, out_features=1)
        >>> logits, attention_list = module(slide, mask=mask)
        >>> tile_scores = module.score_model(slide, mask=mask)

    Parameters
    ----------
    in_features: int
    out_features: int = 1
    emb_dim: Optional[int] = None
    depth: int = 1
    d_model: int = 128
    num_heads: int = 4
    feed_forward_hidden: Optional[List[int]] = None
    dropout: float = 0.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        emb_dim: Optional[int] = None,
        depth: int = 1,
        d_model: int = 128,
        num_heads: int = 4,
        feed_forward_hidden: Optional[List[int]] = None,
        dropout: float = 0.0,
        device: str = "cuda",
        bias_init: Optional[torch.nn.Parameter] = None,
    ):
        super().__init__()

        if emb_dim is not None:
            self.emb_layer = MaskedLinear(
                in_features=input_dim,
                out_features=emb_dim,
                bias=True,
                mask_value=0.0,
            )
            self.emb_dim = emb_dim
            self.enable_embedding = True
        else:
            self.emb_dim = input_dim
            self.enable_embedding = False

        self.transformer = Transformer(
            in_features=self.emb_dim,
            depth=depth,
            d_model=d_model,
            num_heads=num_heads,
            feed_forward_hidden=feed_forward_hidden,
            dropout=dropout,
        )

        self.mlp = MaskedLinear(self.emb_dim, output_dim, mask_value=0.0)
        self.device = device
        self.to(self.device)

        if bias_init is not None:
            self.mlp.bias = bias_init

    def forward(
        self,
        x: torch.Tensor,
        neighbors: torch.LongTensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Compute forward pass.

        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, FEATURES)
        neighbors : torch.Tensor
            (B, SEQ_LEN, K_NEIGHBORS)
        mask: Optional[torch.BoolTensor]
            (B, N_TILES, 1), True for values that were padded.

        Returns
        -------
        logits: torch.Tensor
            (B, OUT_FEATURES)
        """
        # Ensure batch is 3d
        no_batch = x.ndim == 2
        if no_batch:
            x = x[None, :]
            if mask is not None:
                mask = mask[None, :]

        # (Optional) Embedding layer
        if self.enable_embedding:
            x = self.emb_layer(x, mask)

        # Transformer
        x = self.transformer(x, neighbors, mask)

        # Prediction head
        logits = self.mlp(x, mask)

        return logits

    def score_model(
        self,
        x: torch.Tensor,
        neighbors: torch.LongTensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Get prediction logits for each tile.

        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, FEATURES)
        neighbors : torch.Tensor
            (B, SEQ_LEN, K_NEIGHBORS)
        mask: Optional[torch.BoolTensor]
            (B, N_TILES, 1), True for values that were padded.

        Returns
        -------
        attention_per_layer: torch.Tensor
            (B, N_TILES, OUT_FEATURES)
        """
        store_pooling = self.use_pooling
        self.use_pooling = False
        logits = self.forward(x=x, neighbors=neighbors, mask=mask)
        if mask is not None:
            logits = logits.masked_fill(mask, 0.0)
        self.use_pooling = store_pooling
        return logits

    def get_attention(
        self,
        x: torch.Tensor,
        neighbors: torch.LongTensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Get attention maps for each layer.

        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, FEATURES)
        neighbors : torch.Tensor
            (B, SEQ_LEN, K_NEIGHBORS)
        mask: Optional[torch.BoolTensor]
            (B, N_TILES, 1), True for values that were padded.

        Returns
        -------
        attention_list: list[torch.Tensor]
            [(B, NUM_HEADS, N_TILES, N_TILES) * DEPTH]
        """
        # Ensure batch is 3d
        no_batch = x.ndim == 2
        if no_batch:
            x = x[None, :]
            if mask is not None:
                mask = mask[None, :]

        # (Optional) Embedding layer
        if self.enable_embedding:
            x = self.emb_layer(x, mask)

        # Transformer
        attention_list = self.transformer.attention(x, neighbors, mask)

        return attention_list


class Transformer(torch.nn.Module):
    """Transformer architecture.

    Parameters
    ----------
    in_features: int
    depth: int
    d_model: int
    num_heads: int
    feed_forward_hidden: Optional[List[int]]
    dropout: float
    """

    def __init__(
        self,
        in_features: int,
        depth: int,
        d_model: int,
        num_heads: int,
        feed_forward_hidden: Optional[List[int]],
        dropout: float,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        torch.nn.LayerNorm(in_features),
                        LocalSelfAttention(
                            in_features,
                            out_features=in_features,
                            d_model=d_model,
                            num_heads=num_heads,
                            dropout=dropout,
                        ),
                        torch.nn.LayerNorm(in_features),
                        TilesMLP(
                            in_features=in_features,
                            out_features=in_features,
                            hidden=feed_forward_hidden,
                            activation=torch.nn.ReLU(),
                            dropout=torch.nn.Dropout(dropout),
                        ),
                    ]
                )
            )

    def attention(
        self,
        x: torch.Tensor,
        neighbors: torch.LongTensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Get attention maps for each layer.

        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, FEATURES)
        neighbors : torch.Tensor
            (B, SEQ_LEN, K_NEIGHBORS)
        mask: Optional[torch.BoolTensor]
            (B, N_TILES, 1), True for values that were padded.

        Returns
        -------
        attention_per_layer: list[torch.Tensor]
            [(B, NUM_HEADS, N_TILES, N_TILES) * DEPTH]
        """
        attention_per_layer = []
        for norm1, attention_layer, norm2, feed_forward in self.layers:
            x1 = norm1(x)
            scaled_attention = attention_layer.attention(x1, neighbors, mask)
            attention_per_layer.append(scaled_attention)
            x1 = attention_layer(x1, neighbors, mask) + x
            x2 = norm2(x1)
            x = feed_forward(x2, mask) + x1
        return attention_per_layer

    def forward(
        self,
        x: torch.Tensor,
        neighbors: torch.LongTensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Compute forward pass.

        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, FEATURES)
        neighbors : torch.Tensor
            (B, SEQ_LEN, K_NEIGHBORS)
        mask: Optional[torch.BoolTensor]
            (B, N_TILES, 1), True for values that were padded.

        Returns
        -------
        x: tuple[torch.Tensor
            (B, N_TILES, FEATURES)
        """
        for norm1, attention_layer, norm2, feed_forward in self.layers:
            x1 = norm1(x)
            x1 = attention_layer(x1, neighbors, mask) + x
            x2 = norm2(x1)
            x = feed_forward(x2, mask) + x1
        return x
