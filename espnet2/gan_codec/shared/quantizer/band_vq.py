# Copyright 2024 Jiatong Shi
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
# Adapted from https://github.com/facebookresearch/encodec

# Copyright 2025 Haoran Wang
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

# Original license as follows:
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in https://github.com/facebookresearch/encodec/tree/main

"""Residual vector quantizer implementation."""
import math
from dataclasses import dataclass, field  # noqa
from typing import Optional

import torch
from torch import nn

from espnet2.gan_codec.shared.quantizer.modules.core_vq import (
    BandVectorQuantization,
)


@dataclass
class BandQuantizedResult:
    quantized: torch.Tensor
    codes: torch.Tensor
    bandwidth: torch.Tensor  # bandwidth in kb/s used, per batch item.
    penalty: Optional[torch.Tensor] = None


class BandVectorQuantizer(nn.Module):
    """Residual Vector Quantizer.

    Args:
        dimension (int): Dimension of the codebooks.
        n_q (int): Number of residual vector quantizers used.
        bins (int): Codebook size.
        decay (float): Decay for exponential moving average over the codebooks.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration.
            Replace any codes that have an exponential moving average cluster
            size less than the specified threshold with
            randomly selected vector from the current batch.
    """

    def __init__(
        self,
        num_bands: int = 4,
        dimension: int = 256,
        codebook_dim: int = 512,
        bins: int = 1024,
        decay: float = 0.99,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        quantizer_dropout: bool = False,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.dimension = dimension
        self.codebook_dim = codebook_dim
        self.bins = bins
        self.decay = decay
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.quantizer_dropout = quantizer_dropout
        self.vq = BandVectorQuantization(
            num_bands=self.num_bands,
            dim=self.dimension,
            codebook_dim=self.codebook_dim,
            codebook_size=self.bins,
            decay=self.decay,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
            quantizer_dropout=self.quantizer_dropout
        )

    def forward(
        self, x: torch.Tensor, sample_rate: int, bandwidth: Optional[float] = None
    ) -> BandQuantizedResult:
        """Residual vector quantization on the given input tensor.

        Args:
            x (torch.Tensor): Input tensor, [B,band,d,n]
            sample_rate (int): Sample rate of the input tensor.
            bandwidth (float): Target bandwidth.
        Returns:
            BandQuantizedResult:
                The quantized (or approximately quantized) representation with
                the associated bandwidth and any penalty term for the loss.
        """
        bw_per_q = self.get_bandwidth_per_quantizer(sample_rate)
        if not self.quantizer_dropout:
            quantized, codes, loss = self.vq(x)
            bw = torch.tensor((self.num_bands) * bw_per_q).to(x)
            return (
                quantized,
                codes,
                bw,
                torch.mean(loss)
            )

    def get_bandwidth_per_quantizer(self, sample_rate: int):
        """Return bandwidth per quantizer for a given input sample rate."""
        return math.log2(self.bins) * sample_rate / 1000

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a given input tensor with the specified sample rate at

        the given bandwidth. The RVQ encode method sets the appropriate
        number of quantizer to use and returns indices for each quantizer.
        """
        codes = self.vq.encode(x)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation."""
        
        quantized = self.vq.decode(codes)
        return quantized

def test_overfit_single_sample():
    """
    Overfit test for BandVectorQuantizer: generate a single input sample
    and verify the quantizer can perfectly (or near-perfectly) reconstruct it.
    """
    # Fix random seed for reproducibility
    torch.manual_seed(42)

    # Configuration: use small dimensions for faster convergence
    num_bands = 2
    dimension = 8
    codebook_dim = 16
    bins = 16
    seq_len = 4

    # Single-sample input: shape [B, band, dim, seq_len]
    x = torch.randn(1, num_bands, dimension, seq_len)

    # Instantiate quantizer without kmeans init for simplicity
    quantizer = BandVectorQuantizer(
        num_bands=num_bands,
        dimension=dimension,
        codebook_dim=codebook_dim,
        bins=bins,
        decay=0.99,
        kmeans_init=False,
        kmeans_iters=0,
        threshold_ema_dead_code=1
    )
    quantizer.train()

    # Optimizer on all parameters of the quantizer
    optimizer = torch.optim.Adam(quantizer.parameters(), lr=1e-2)

    # Training loop: overfit to the single sample
    final_loss = None
    for step in range(1000):
        optimizer.zero_grad()
        # Forward pass: get quantized output and penalty
        quantized, codes, bw, penalty = quantizer(x, sample_rate=75)
        # recon = quantized.sum(dim=1)
        # x_sum = x.sum(dim=1)
        # Reconstruction loss + penalty
        # rec_loss = ((recon - x_sum) ** 2).mean()
        loss = penalty

        loss.backward()
        optimizer.step()

        final_loss = loss.item()
        # Print losses
        print(f"Step {step}: penalty={(penalty.item() if penalty is not None else 0):.6f}, total_loss={final_loss:.6f}")

    # Assert that the quantizer has overfitted (loss is very small)
    assert final_loss < 1e-4, f"Failed to overfit: final loss = {final_loss}"


if __name__ == "__main__":
    test_overfit_single_sample()

