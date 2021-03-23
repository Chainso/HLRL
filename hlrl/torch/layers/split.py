from typing import Any, Tuple

import torch
import torch.nn as nn

class SplitLayer(nn.Linear):
    """
    A linear layer with a dense and a split component.
    """
    def __init__(
            self,
            dense_features: int,
            split_space: Tuple[int, ...],
            out_features: int,
            *args: Any,
            **kwargs: Any
        ) -> None:
        """
        Creates the split layer with dense inputs being repeated along the split
        space.

        Args:
            dense_features: The number of dense input features.
            split_space: The tuple space of the split input features.
            out_features: The number of output features.
            args: Additional positional arguments for the linear layer.
            kwargs: Additional keyword arguments for the linear layer.

        Attributes:
            dense_weight: The learnable weights for the dense features.
            split_weight: The learnable weights for the split features.
            bias: The learnable bias of the layer.
        """
        self.dense_features = dense_features
        self.split_space = split_space
        self.split_features = sum(split_space)

        super().__init__(
            dense_features + self.split_features, out_features, *args, **kwargs
        )

    def forward(
            self,
            dense_input: torch.Tensor,
            split_input: torch.Tensor
        ) -> torch.Tensor:
        """
        Does a forward pass on the concatenated and repeated dense input and
        split inputs.

        Args:
            dense_input: The inputs for the dense features.
            split_input: The inputs for the split features.

        Returns:
            The forward pass on the inputs.
        """
        linear_input = self.make_split_linear_input(dense_input, split_input)

        return super().forward(linear_input)

    def make_split_linear_input(
            self,
            dense_input: torch.Tensor,
            split_input: torch.Tensor
        ) -> torch.Tensor:
        """
        Creates the split input for the layer.

        Args:
            dense_input: The inputs for the dense features.
            split_input: The inputs for the split features.

        Returns:
            Returns the dense inputs concatenated and repeated with each split
            input.
        """
        batch_size = dense_input.shape[0]
        device = dense_input.device

        dense_repeated = dense_input.repeat_interleave(
            len(self.split_space), dim=0
        )

        single_scatter_idxs = torch.arange(len(self.split_space), device=device)
        single_scatter_idxs = single_scatter_idxs.repeat_interleave(
            self.split_space, dim=0
        )
        single_scatter_idxs = single_scatter_idxs.unsqueeze(0).expand(
            batch_size, -1
        )

        scatter_offsets = torch.arange(batch_size, device=device)
        scatter_offsets *= len(self.split_space)
        scatter_offsets = scatter_offsets.unsqueeze(-1)

        scatter_idxs = scatter_offsets + single_scatter_idxs

        split_cat = torch.zeros(
            batch_size * len(self.split_space), self.split_features,
            device=device
        )

        split_cat = split_cat.scatter(0, scatter_idxs, split_input)

        linear_input = torch.cat([dense_repeated, split_cat], dim=-1)

        return linear_input

    def extra_repr(self) -> str:
        """
        A more detailed representation of the split layer.

        Returns:
            The detailed representation of the split layer.
        """
        linear_repr = super().extra_repr()

        additional_repr = "dense_features={}, split_space={}".format(
            self.dense_features, self.split_space
        )

        return linear_repr + " " + addition_repr
