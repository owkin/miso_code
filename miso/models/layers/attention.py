"""Implementation of self-attention and local self-attention."""

import math
import torch
import warnings

from typing import Optional, List, Union


class SelfAttention(torch.nn.Module):
    """Self-attention module, as defined in https://arxiv.org/abs/1706.03762.

    Parameters
    ----------
    in_features: int
    d_model: int
    num_heads: int
        Number of attention heads.
        d_model must be divisible by num_heads.
    dropout: float
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.head_dim = d_model // num_heads
        self.num_heads = num_heads
        self.dropout = dropout

        self.scale = self.head_dim**-0.5

        self.to_qkv = MaskedLinear(in_features, d_model * 3, mask_value=0.0, bias=True)
        self.to_out = MaskedLinear(d_model, out_features, mask_value=0.0, bias=False)

    def attention(
        self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        """Compute the attention maps for interpretability.

        To do so we pass as the value v an identity matrix of size (B, H, N, N).

        Parameters
        ----------
        x: torch.Tensor
            (B, SEQ_LEN, IN_FEATURES)
        mask: Optional[torch.BoolTensor]
            (B, SEQ_LEN, 1), True for values that were padded.

        Returns
        -------
        scaled_attention: torch.Tensor
            (B, NUM_HEADS, N_TILES, N_TILES)
        """
        qkv = self.to_qkv(x, mask=mask).chunk(3, dim=-1)  # (B, SEQ_LEN, D_MODEL, 3)

        batch_size, seq_len, _ = x.shape
        q, k, _ = map(
            lambda t: torch.permute(
                t.view(batch_size, seq_len, self.num_heads, self.head_dim), [0, 2, 1, 3]
            ),
            qkv,
        )  # 3 * (B, NUM_HEADS, SEQ_LEN, HEAD_DIM)

        dummy_v = (
            torch.eye(seq_len)
            .view(1, 1, seq_len, seq_len)
            .expand(batch_size, self.num_heads, seq_len, seq_len)
            .to(x.device)
        )

        if mask is not None:
            mask_mat = (
                mask.type(torch.float) @ mask.permute(0, 2, 1).type(torch.float)
            ).type(torch.bool)  # (B, SEQ_LEN, SEQ_LEN)
            mask_mat = mask_mat.unsqueeze(1).expand(
                [-1, self.num_heads, -1, -1]
            )  # (B, NUM_HEADS, SEQ_LEN, SEQ_LEN)
            mask_mat = ~mask_mat  # mask has to be inverted due to the implementation
        else:
            mask_mat = None

        if int(torch.__version__[0]) > 1:
            with torch.backends.cuda.sdp_kernel():
                scaled_attention = torch.nn.functional.scaled_dot_product_attention(
                    q, k, dummy_v, mask_mat, dropout_p=self.dropout
                )  # (B, NUM_HEADS, SEQ_LEN, SEQ_LEN)
        else:
            warnings.warn("Self attention is faster with torch v2")
            # code that makes the same thing as `scaled_dot_product_attention`
            attn_weight = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
            if mask_mat is not None:
                mask_mat = mask_mat.masked_fill(~mask_mat, -float("inf"))
                attn_weight = attn_weight + mask_mat
            attn_weight = torch.softmax(attn_weight, dim=-1)
            attn_weight = torch.nn.functional.dropout(attn_weight, self.dropout)
            scaled_attention = attn_weight @ dummy_v

        return scaled_attention

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        """Compute forward pass.

        Parameters
        ----------
        x: torch.Tensor
            (B, SEQ_LEN, IN_FEATURES)
        mask: Optional[torch.BoolTensor]
            (B, SEQ_LEN, 1), True for values that were padded.

        Returns
        -------
        scaled_attention: torch.Tensor
            (B, SEQ_LEN, IN_FEATURES)
        """
        qkv = self.to_qkv(x, mask=mask).chunk(3, dim=-1)  # (B, SEQ_LEN, D_MODEL, 3)

        batch_size, seq_len, _ = x.shape
        q, k, v = map(
            lambda t: torch.permute(
                t.view(batch_size, seq_len, self.num_heads, self.head_dim), [0, 2, 1, 3]
            ),
            qkv,
        )  # 3 * (B, NUM_HEADS, SEQ_LEN, HEAD_DIM)

        if mask is not None:
            mask_mat = (
                mask.type(torch.float) @ mask.permute(0, 2, 1).type(torch.float)
            ).type(torch.bool)  # (B, SEQ_LEN, SEQ_LEN)
            mask_mat = mask_mat.unsqueeze(1).expand(
                [-1, self.num_heads, -1, -1]
            )  # (B, NUM_HEADS, SEQ_LEN, SEQ_LEN)
            mask_mat = ~mask_mat  # mask has to be inverted due to the implementation
        else:
            mask_mat = None

        if int(torch.__version__[0]) > 1:
            with torch.backends.cuda.sdp_kernel():
                scaled_x = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, mask_mat, dropout_p=self.dropout
                )  # (B, NUM_HEADS, SEQ_LEN, HEAD_DIM)
        else:
            warnings.warn("Self attention is faster with torch v2")
            # code that makes the same thing as `scaled_dot_product_attention`
            attn_weight = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
            if mask_mat is not None:
                mask_mat = mask_mat.masked_fill(~mask_mat, -float("inf"))
                attn_weight = attn_weight + mask_mat
            attn_weight = torch.softmax(attn_weight, dim=-1)
            attn_weight = torch.nn.functional.dropout(attn_weight, self.dropout)
            scaled_x = attn_weight @ v

        scaled_x = torch.permute(scaled_x, [0, 2, 1, 3]).reshape(
            batch_size, seq_len, self.num_heads * self.head_dim
        )  # (B, SEQ_LEN, D_MODEL)

        scaled_x = self.to_out(scaled_x, mask=mask)  # (B, SEQ_LEN, IN_FEATURES)

        return scaled_x


class LocalSelfAttention(SelfAttention):
    """Local self-attention module, as defined in https://arxiv.org/abs/2205.06672.

    Parameters
    ----------
    in_features: int
    d_model: int
    num_heads: int
        Number of attention heads.
        d_model must be divisible by num_heads.
    dropout: float
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__(in_features, out_features, d_model, num_heads, dropout)

        self.dropout_func = torch.nn.Dropout(self.dropout)

    def _select_neighbors(self, x: torch.Tensor, neighbors: torch.LongTensor):
        """Select neighbors in a 4D tensor.

        Parameters
        ----------
        x: torch.Tensor
            (B, NUM_HEADS, SEQ_LEN, DIM)
        neighbors : torch.Tensor
            (B * NUM_HEADS * SEQ_LEN, K_NEIGHBORS)

        Returns
        -------
        x_neighbors: torch.Tensor
            (B, NUM_HEADS, SEQ_LEN, K_NEIGHBORS, DIM)
        """
        batch_size, num_heads, seq_len, dim = x.shape
        _, k_neighbors = neighbors.shape
        x = x.reshape(batch_size * num_heads * seq_len, dim)
        x_neighbors = x[neighbors]
        x_neighbors = x_neighbors.reshape(
            batch_size, num_heads, seq_len, k_neighbors, dim
        )
        return x_neighbors

    def attention(
        self,
        x: torch.Tensor,
        neighbors: torch.LongTensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Compute the attention maps for interpretability.

        To do so we pass as the value v an identity matrix of size (B, H, N, N).

        Parameters
        ----------
        x: torch.Tensor
            (B, SEQ_LEN, IN_FEATURES)
        neighbors : torch.Tensor
            (B, SEQ_LEN, K_NEIGHBORS)
        mask: Optional[torch.BoolTensor]
            (B, SEQ_LEN, 1), True for values that were padded.

        Returns
        -------
        attn_weight: torch.Tensor
            (B, NUM_HEADS, SEQ_LEN, SEQ_LEN)
        """
        qkv = self.to_qkv(x, mask=mask).chunk(3, dim=-1)  # (B, SEQ_LEN, D_MODEL, 3)

        batch_size, seq_len, _ = x.shape
        q, k, _ = map(
            lambda t: torch.permute(
                t.view(batch_size, seq_len, self.num_heads, self.head_dim), [0, 2, 1, 3]
            ),
            qkv,
        )  # 3 * (B, NUM_HEADS, SEQ_LEN, HEAD_DIM)

        # Expanding neighbors
        _, _, k_neighbors = neighbors.shape
        neighbors = (
            neighbors.unsqueeze(1).expand(
                batch_size, self.num_heads, seq_len, k_neighbors
            )
            + torch.arange(self.num_heads)[None, :, None, None].to(neighbors.device)
            * seq_len
            + torch.arange(batch_size)[:, None, None, None].to(neighbors.device)
            * seq_len
            * self.num_heads
        )
        neighbors = neighbors.reshape(
            batch_size * self.num_heads * seq_len, k_neighbors
        )

        # Expanding mask
        if mask is not None:
            mask_mat = mask.unsqueeze(1).expand(batch_size, self.num_heads, seq_len, 1)
            mask_mat = self._select_neighbors(mask_mat, neighbors).squeeze(-1)
            mask_mat = mask_mat.float().masked_fill(mask_mat, -float("inf"))
        else:
            mask_mat = None

        # Selecting neighbors for keys and values
        k = self._select_neighbors(k, neighbors)

        # Computing attention matrix and scaled input
        attn_weight = torch.einsum("b h n d, b h n k d -> b h n k", q, k) * self.scale
        if mask_mat is not None:
            attn_weight = attn_weight + mask_mat
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = self.dropout_func(attn_weight)
        if mask is not None:
            attn_weight = attn_weight.masked_fill(mask.unsqueeze(1), 0)

        return attn_weight

    def forward(
        self,
        x: torch.LongTensor,
        neighbors: torch.LongTensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Compute forward pass.

        Parameters
        ----------
        x: torch.Tensor
            (B, SEQ_LEN, IN_FEATURES)
        neighbors : torch.Tensor
            (B, SEQ_LEN, K_NEIGHBORS)
        mask: Optional[torch.BoolTensor]
            (B, SEQ_LEN, 1), True for values that were padded.

        Returns
        -------
        scaled_attention: torch.Tensor
            (B, SEQ_LEN, IN_FEATURES)
        """
        qkv = self.to_qkv(x, mask=mask).chunk(3, dim=-1)  # (B, SEQ_LEN, D_MODEL, 3)

        batch_size, seq_len, _ = x.shape
        q, k, v = map(
            lambda t: torch.permute(
                t.view(batch_size, seq_len, self.num_heads, self.head_dim), [0, 2, 1, 3]
            ),
            qkv,
        )  # 3 * (B, NUM_HEADS, SEQ_LEN, HEAD_DIM)

        # Expanding neighbors
        _, _, k_neighbors = neighbors.shape
        neighbors = (
            neighbors.unsqueeze(1).expand(
                batch_size, self.num_heads, seq_len, k_neighbors
            )
            + torch.arange(self.num_heads)[None, :, None, None].to(neighbors.device)
            * seq_len
            + torch.arange(batch_size)[:, None, None, None].to(neighbors.device)
            * seq_len
            * self.num_heads
        )
        neighbors = neighbors.reshape(
            batch_size * self.num_heads * seq_len, k_neighbors
        )

        # Expanding mask
        if mask is not None:
            mask_mat = mask.unsqueeze(1).expand(batch_size, self.num_heads, seq_len, 1)
            mask_mat = self._select_neighbors(mask_mat, neighbors).squeeze(-1)
            mask_mat = mask_mat.float().masked_fill(mask_mat, -float("inf"))
        else:
            mask_mat = None

        # Selecting neighbors for keys and values
        k = self._select_neighbors(k, neighbors)
        v = self._select_neighbors(v, neighbors)

        # Computing attention matrix and scaled input
        attn_weight = torch.einsum("b h n d, b h n k d -> b h n k", q, k) * self.scale
        if mask_mat is not None and attn_weight.shape[-1] > 1:
            attn_weight = attn_weight + mask_mat
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = self.dropout_func(attn_weight)
        scaled_x = torch.einsum("b h n k, b h n k d -> b h n d", attn_weight, v)

        scaled_x = torch.permute(scaled_x, [0, 2, 1, 3]).reshape(
            batch_size, seq_len, self.num_heads * self.head_dim
        )  # (B, SEQ_LEN, D_MODEL)

        scaled_x = self.to_out(scaled_x, mask=mask)  # (B, SEQ_LEN, IN_FEATURES)
        if mask is not None:
            scaled_x = scaled_x.masked_fill(mask, 0)

        return scaled_x


class MaskedLinear(torch.nn.Linear):
    """
    Linear layer to be applied tile wise.
    This layer can be used in combination with a mask
    to prevent padding tiles from influencing the values of a subsequent
    activation.

    Example:
        >>> module = Linear(in_features=128, out_features=1) # With Linear
        >>> out = module(slide)
        >>> wrong_value = torch.sigmoid(out) # Value is influenced by padding
        >>> module = MaskedLinear(in_features=128, out_features=1, mask_value='-inf') # With MaskedLinear
        >>> out = module(slide, mask) # Padding now has the '-inf' value
        >>> correct_value = torch.sigmoid(out) # Value is not influenced by padding as sigmoid('-inf') = 0


    Parameters
    ----------
    in_features: int
        size of each input sample
    out_features: int
        size of each output sample
    mask_value: Union[str, int]
        value to give to the mask
    bias: bool = True
        If set to ``False``, the layer will not learn an additive bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask_value: Union[str, float],
        bias: bool = True,
    ):
        super(MaskedLinear, self).__init__(
            in_features=in_features, out_features=out_features, bias=bias
        )
        self.mask_value = mask_value

    def forward(self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None):
        """
        Parameters
        ----------
        x: torch.Tensor
            (B, SEQ_LEN, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, SEQ_LEN, 1), True for values that were padded.

        Returns
        -------
        x: torch.Tensor
            (B, SEQ_LEN, OUT_FEATURES)
        """
        x = super(MaskedLinear, self).forward(x)
        if mask is not None:
            x = x.masked_fill(mask, float(self.mask_value))
        return x

    def extra_repr(self):
        return "in_features={}, out_features={}, mask_value={}, bias={}".format(
            self.in_features, self.out_features, self.mask_value, self.bias is not None
        )


class TilesMLP(torch.nn.Module):
    """
    MLP to be applied to tiles to compute scores.
    This module can be used in combination of a mask
    to prevent padding from influencing the scores values.

    Parameters
    ----------
    in_features: int
        size of each input sample
    out_features: int
        size of each output sample
    hidden: Optional[List[int]] = None:
        Number of hidden layers and their respective number of features.
    bias: bool = True
        If set to ``False``, the layer will not learn an additive bias.
    activation: torch.nn.Module = torch.nn.Sigmoid()
        MLP activation function
    dropout: Optional[torch.nn.Module] = None
        Optional dropout module. Will be interlaced with the linear layers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        hidden: Optional[List[int]] = None,
        bias: bool = True,
        activation: torch.nn.Module = torch.nn.Sigmoid(),
        dropout: Optional[torch.nn.Module] = None,
    ):
        super(TilesMLP, self).__init__()

        self.hidden_layers = torch.nn.ModuleList()
        if hidden is not None:
            for h in hidden:
                self.hidden_layers.append(
                    MaskedLinear(in_features, h, bias=bias, mask_value="-inf")
                )
                self.hidden_layers.append(activation)
                if dropout:
                    self.hidden_layers.append(dropout)
                in_features = h

        self.hidden_layers.append(torch.nn.Linear(in_features, out_features, bias=bias))

    def forward(self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None):
        """
        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, N_TILES), True for values that were padded.

        Returns
        -------
        x: torch.Tensor
            (B, N_TILES, OUT_FEATURES)
        """
        for layer in self.hidden_layers:
            if isinstance(layer, MaskedLinear):
                x = layer(x, mask)
            else:
                x = layer(x)
        return x
