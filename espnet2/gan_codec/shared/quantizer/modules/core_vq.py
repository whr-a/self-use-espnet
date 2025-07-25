# Copyright 2024 Jiatong Shi
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
# Adapted from https://github.com/facebookresearch/encodec
# Original license as follows:
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This implementation is inspired from
# https://github.com/lucidrains/vector-quantize-pytorch
# which is released under MIT License. Hereafter, the original license:
# MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Core vector quantization implementation."""
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from espnet2.gan_codec.shared.quantizer.modules.distrib import broadcast_tensors


def default(val: Any, d: Any) -> Any:
    return val if val is not None else d


def ema_inplace(moving_avg, new, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories: int, epsilon: float = 1e-5):
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples, num_clusters: int, num_iters: int = 10):
    dim, dtype = samples.shape[-1], samples.dtype

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        diffs = rearrange(samples, "n d -> n () d") - rearrange(means, "c d -> () c d")
        dists = -(diffs**2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        means = torch.where(zero_mask[..., None], means, new_means)

    # Cluster centroids and number of frames per cluster
    return means, bins


class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance.

    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
            If set to true, run the k-means algorithm on the first training batch
            and use the learned centroids as initialization.
        kmeans_iters (int): Number of iterations used for k-means algorithm at
            initialization.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_ema_dead_code (int): Threshold for dead code expiration.
            Replace any codes that have an exponential moving average cluster size
            less than the specified threshold with randomly selected vector from
            the current batch.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: int = False,
        kmeans_iters: int = 10,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.decay = decay
        init_fn: Union[Callable[..., torch.Tensor], Any] = (
            uniform_init if not kmeans_init else torch.zeros
        )
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size

        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.inited:
            return

        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))
        # Make sure all buffers across workers are in sync after initialization
        broadcast_tensors(self.buffers())

    def replace_(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)
        broadcast_tensors(self.buffers())

    def preprocess(self, x):
        x = rearrange(x, "... d -> (...) d")
        return x

    def quantize(self, x):
        embed = self.embed.t()
        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    def postprocess_emb(self, embed_ind, shape):
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind):
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x):
        shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x):
        """Codebook Forward with EMA.

        Args:
            x (Tensor): Vector for quantization (B, T, D)

        Return:
            Tensor: Quantized output (B, T, D)
            Tensor: Codebook Index (B, T)
        """
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x)  # (BxT, D)

        # Initialize the embedding (only activated for the first time)
        self.init_embed_(x)

        # Quantization Process
        embed_ind = self.quantize(x)  # (BxT)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)  # (BxT, V)
        embed_ind = self.postprocess_emb(embed_ind, shape)  # (B, T)
        quantize = self.dequantize(embed_ind)  # (B, T, D)

        if self.training:
            # We do the expiry of code at that point as buffers are in sync
            # and all the workers will take the same decision.
            self.expire_codes_(x)

            # ema update number of frames per cluster
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)

            # Use encoder embedding to update ema with assignments
            embed_sum = x.t() @ embed_onehot  # (D, BxT) @ (BxT, V) -> (D, V)

            # ema udpate embedding
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
                * self.cluster_size.sum()
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

        return quantize, embed_ind


class VectorQuantization(nn.Module):
    """Vector quantization implementation.

    Currently supports only euclidean distance.
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
        codebook_dim (int): Codebook dimension. If not defined, uses the specified
            dimension in dim.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration.
            Replace any codes that have an exponential moving average cluster size
            less than the specified threshold with randomly selected vector from
            the current batch.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: Optional[int] = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        commitment_weight: float = 1.0,
        quantizer_dropout: bool = False,
    ):
        super().__init__()
        _codebook_dim: int = default(codebook_dim, dim)

        requires_projection = _codebook_dim != dim
        self.project_in = (
            nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity()
        )
        self.project_out = (
            nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity()
        )

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight

        self._codebook = EuclideanCodebook(
            dim=_codebook_dim,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            epsilon=epsilon,
            threshold_ema_dead_code=threshold_ema_dead_code,
        )
        self.codebook_size = codebook_size
        self.quantizer_dropout = quantizer_dropout

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x):
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

    def forward(self, x, mask=None):
        device = x.device
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)

        quantize, embed_ind = self._codebook(x)

        if self.training:
            quantize = x + (quantize - x).detach()

        if not self.quantizer_dropout:
            loss = torch.tensor([0.0], device=device, requires_grad=self.training)

            if self.training:
                commit_loss = F.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss

            quantize = self.project_out(quantize)
            quantize = rearrange(quantize, "b n d -> b d n")
            return quantize, embed_ind, loss
        else:
            commit_loss = torch.tensor(
                [0.0], device=device, requires_grad=self.training
            )
            quant_loss = torch.tensor([0.0], device=device, requires_grad=self.training)
            if self.training:
                if self.quantizer_dropout:
                    _commit_loss = F.mse_loss(
                        quantize.detach(), x, reduction="none"
                    ).mean([1, 2])
                    commit_loss = commit_loss + (_commit_loss * mask).mean()
                    _quant_loss = F.mse_loss(
                        quantize, x.detach(), reduction="none"
                    ).mean([1, 2])
                    quant_loss = quant_loss + (_quant_loss * mask).mean()

                else:
                    _commit_loss = F.mse_loss(quantize.detach(), x)
                    commit_loss = commit_loss + _commit_loss
                    _quant_loss = F.mse_loss(quantize, x.detach(), reduction="none")
                    quant_loss = quant_loss + _quant_loss

            quantize = self.project_out(quantize)
            quantize = rearrange(quantize, "b n d -> b d n")
            return quantize, embed_ind, commit_loss, quant_loss


class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation.

    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, *, num_quantizers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )
        self.quantizer_dropout = kwargs.get("quantizer_dropout")

    def forward(self, x, n_q: Optional[int] = None):
        quantized_out = 0.0
        residual = x

        if not self.quantizer_dropout:
            all_losses = []
            all_indices = []

            n_q = n_q or len(self.layers)

            for layer in self.layers[:n_q]:
                quantized, indices, loss = layer(residual)
                residual = residual - quantized
                quantized_out = quantized_out + quantized

                all_indices.append(indices)
                all_losses.append(loss)

            if self.training:
                # Solving subtle bug with STE and RVQ
                # For more, https://github.com/facebookresearch/encodec/issues/25
                quantized_out = x + (quantized_out - x).detach()

            out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
            return quantized_out, out_indices, out_losses
        else:
            all_commit_losses = []
            all_quant_losses = []
            all_indices = []

            n_q = n_q or len(self.layers)
            if self.training:
                n_q = torch.ones((x.shape[0],)) * len(self.layers) + 1
                dropout = torch.randint(1, len(self.layers) + 1, (x.shape[0],))
                n_dropout = int(x.shape[0] * self.quantizer_dropout)
                n_q[:n_dropout] = dropout[:n_dropout]
                n_q = n_q.to(x.device)

            for i, layer in enumerate(self.layers):
                if self.training is False and i >= n_q:
                    break
                mask = torch.full((x.shape[0],), fill_value=i, device=x.device) < n_q
                quantized, indices, commit_loss, quant_loss = layer(residual, mask)
                residual = residual - quantized
                quantized_out = quantized_out + quantized * mask[:, None, None]

                all_indices.append(indices)
                all_commit_losses.append(commit_loss)
                all_quant_losses.append(quant_loss)

            if self.training:
                # Solving subtle bug with STE and RVQ
                # For more, https://github.com/facebookresearch/encodec/issues/25
                quantized_out = x + (quantized_out - x).detach()

            out_commit_losses, out_quant_losses, out_indices = map(
                torch.stack, (all_commit_losses, all_quant_losses, all_indices)
            )
            return quantized_out, out_indices, out_commit_losses, out_quant_losses

    def encode(
        self, x: torch.Tensor, n_q: Optional[int] = None, st: Optional[int] = None
    ) -> torch.Tensor:
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        st = st or 0
        for layer in self.layers[st:n_q]:  # 设置解码的起止layer
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out
class BandVectorQuantization(nn.Module):
    """Residual vector quantization implementation.

    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, *, num_bands: int, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_bands)]
        )
        self.num_bands = num_bands
        self.quantizer_dropout = kwargs.get("quantizer_dropout")

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of shape [B, bands, d]
        Returns:
            quantized_per: Tensor [B, bands+1, d]  ← 每个 band+最后残差 的量化输出
            all_indices: LongTensor [bands+1, B, ...]
            all_losses:  Tensor [bands+1, B, ...]  （如果 VectorQuantization 返回 loss）
        """
        B, bands, d, n = x.shape
        assert bands == self.num_bands, "输入 bands 必须和初始化时一致"

        if not self.quantizer_dropout:
            residual = torch.zeros((B, d, n), device=x.device)
            quantized_per = []   # 存每层量化后的 [B, d, n]
            all_indices = []
            all_loss = []

            # 对前 bands 个 band 做 VQ
            for i in range(bands):
                inp = residual + x[:, i, :, :]
                quantized, indices, loss = self.layers[i](inp)
                residual = inp - quantized

                quantized_per.append(quantized)
                all_indices.append(indices)
                all_loss.append(loss)

            # stack → quantized_per: [B, bands+1, d]
            quantized_per = torch.stack(quantized_per, dim=1)
            # stack → all_indices, all_losses: [bands+1, B, ...]
            all_indices = torch.stack(all_indices, dim=1)
            all_loss = torch.stack(all_loss)

            return quantized_per, all_indices, all_loss

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor [B, bands, d]
        Returns:
            indices: Tensor [bands+1, B, *]
        """
        B, bands, d, n = x.shape
        assert bands == self.num_bands
        residual = torch.zeros((B, d, n), device=x.device)   # 【MOD】初始残差 = 0
        all_indices = []

        # 1. 对前 bands 个 signal band 做 VQ
        for i in range(bands):
            inp = residual + x[:, i, :, :]                  # 【MOD】残差+第 i 个 band
            idx = self.layers[i].encode(inp)
            quantized = self.layers[i].decode(idx)
            residual = inp - quantized                     # 【MOD】更新残差
            all_indices.append(idx)

        # stack → [bands+11, B, ...]
        return torch.stack(all_indices, dim=1)

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_indices: LongTensor [bands+1, B, ...]
        Returns:
            quant_per_band: Tensor [B, bands+1, d]
        """

        recon_per = []
        B, bands, n = q_indices.shape
        for i in range(bands):
            quant_i = self.layers[i].decode(q_indices[:,i,:])
            recon_per.append(quant_i)
        return torch.stack(recon_per, dim=1)