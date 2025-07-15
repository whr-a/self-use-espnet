import math
import random
from typing import Any, Dict, List, Optional
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from typeguard import typechecked
import functools
from espnet2.gan_codec.shared.decoder.torchcfm.conditional_flow_matching import ConditionalFlowMatcher
import torch.optim as optim

class VQCode2MelCFM_cnn_cfm(nn.Module):
    @typechecked
    def __init__(
        self,
        emb_dim: int = 128,
        conv_channels: int = 256,
        num_conv_layers: int = 3,
        mel_dim: int = 100,
        fs: int = 24000,
        vq_fr: float = 75.0,
        mel_hop_size: int = 256,
        cfm_sigma: float = 0.0,
        flow_hidden_dim: int = 256,
    ):
        super().__init__()
        # VQCode2Mel components
        self.fs = fs
        self.vq_fr = vq_fr
        self.mel_hop_size = mel_hop_size
        convs = []
        for i in range(num_conv_layers):
            in_ch = emb_dim if i == 0 else conv_channels
            convs.append(nn.Conv1d(in_ch, conv_channels, kernel_size=3, padding=1))
            convs.append(nn.ReLU())
        self.conv_layers = nn.Sequential(*convs)
        self.mlp_head = nn.Linear(conv_channels, mel_dim)

        # CFM matcher
        self.matcher = ConditionalFlowMatcher(sigma=cfm_sigma)

        # Flow prediction head\        # input dim: mel_dim + emb_dim + 1 (time)
        self.flow_head = nn.Sequential(
            nn.Linear(mel_dim + emb_dim + 1, flow_hidden_dim),
            nn.ReLU(),
            nn.Linear(flow_hidden_dim, mel_dim),
        )

    def _vq_forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embedding to mel prediction
        x = x.transpose(1, 2)                      # [B, E, T_vq]
        # target_len = int(round(x.shape[-1] / self.vq_fr * self.mel_fr))
        target_len = math.floor(self.fs * x.shape[-1] / self.vq_fr) // self.mel_hop_size
        x = F.interpolate(x, size=target_len, mode='linear', align_corners=True)
        x = x.transpose(1, 2)                      # [B, T_mel, E]
        x = x.transpose(1, 2)                      # [B, E, T_mel]
        x = self.conv_layers(x)                    # [B, C, T_mel]
        x = x.transpose(1, 2)                      # [B, T_mel, C]
        mel = self.mlp_head(x)                     # [B, T_mel, mel_dim]
        return mel

    def _predict_flow(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor
    ) -> torch.Tensor:
        # cond: [B, T_vq, E] -> resize to [B, T_mel, E]
        if cond.shape[1] != x_t.shape[1]:
            cond = cond.transpose(1,2)                                # [B, E, T_vq]
            cond = F.interpolate(cond, size=x_t.shape[1], mode='linear', align_corners=True)
            cond = cond.transpose(1,2)                                # [B, T_mel, E]
        # time feature
        t_feat = t.view(-1, 1).expand(-1, x_t.shape[1]).unsqueeze(-1)  # [B, T_mel, 1]
        inp = torch.cat([x_t, cond, t_feat], dim=-1)                   # [B, T_mel, mel+E+1]
        return self.flow_head(inp)                                     # [B, T_mel, mel_dim]

    def forward(
        self,
        quantized: torch.Tensor, # [B, dim, T]
        mel_gt: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            vq_code: [B, T_vq] int64  
            mel_gt:  [B, T_mel, mel_dim] float
        Returns:
            mel_pred: [B, T_mel, mel_dim]
            total_loss: scalar tensor (recon + cfm)
        """
        quantized = quantized.transpose(1, 2)
        
        # 1. VQ -> initial mel prediction
        mel_pred = self._vq_forward(quantized)
        mel_gt = mel_gt.transpose(1, 2)
        # 2. Build CFM endpoints
        x0, x1 = mel_pred, mel_gt

        # 3. Sample (t, x_t, u_t)
        t, x_t, u_t = self.matcher.sample_location_and_conditional_flow(x0, x1)

        # 5. Predict vector field and compute CFM loss
        v_pred = self._predict_flow(x_t, t, quantized)
        cfm_loss = F.mse_loss(v_pred, u_t)

        # 6. Reconstruction loss
        recon_loss = F.l1_loss(mel_pred, mel_gt)

        total_loss = recon_loss + cfm_loss

        return total_loss
    @torch.no_grad()
    def decode(
        self,
        quantized: torch.Tensor,
        num_steps: int = 50
    ) -> torch.Tensor:
        """
        基于训练好的 CFM 流场，对初始 mel_pred 做 num_steps 步欧拉迭代精修。

        Args:
            vq_code: [B, T_vq] Tensor 输入的 VQ code 序列。
            num_steps: int 迭代步数，越大精修越精细，但越慢。
        Returns:
            x: [B, T_mel, mel_dim] FloatTensor，精修后的 Mel 频谱。
        """
        self.eval()
        quantized = quantized.transpose(1, 2)
        # 1) 获得初始预测 x0
        x = self._vq_forward(quantized)

        if num_steps < 2:
            # t = 0 时一次完整修正
            t = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            v_pred = self._predict_flow(x, t, quantized)
            return x + v_pred

        dt = 1.0 / (num_steps - 1)
        # 3) 在 [0,1] 时间区间上做欧拉积分
        for i in range(num_steps):
            t = torch.full(
                (x.size(0),),
                float(i) * dt,
                device=x.device,
                dtype=x.dtype
            )
            # 预测当前时刻的速度场
            v_pred = self._predict_flow(x, t, quantized)
            # 欧拉更新
            x = x + v_pred * dt

        return x
    def train_decode(
        self,
        quantized: torch.Tensor,
        num_steps: int = 50
    ) -> torch.Tensor:
        """
        基于训练好的 CFM 流场，对初始 mel_pred 做 num_steps 步欧拉迭代精修。

        Args:
            vq_code: [B, T_vq] Tensor 输入的 VQ code 序列。
            num_steps: int 迭代步数，越大精修越精细，但越慢。
        Returns:
            x: [B, T_mel, mel_dim] FloatTensor，精修后的 Mel 频谱。
        """
        quantized = quantized.transpose(1, 2)
        # 1) 获得初始预测 x0
        x = self._vq_forward(quantized)

        if num_steps < 2:
            # t = 0 时一次完整修正
            t = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            v_pred = self._predict_flow(x, t, quantized)
            return x + v_pred

        dt = 1.0 / (num_steps - 1)
        # 3) 在 [0,1] 时间区间上做欧拉积分
        for i in range(num_steps):
            t = torch.full(
                (x.size(0),),
                float(i) * dt,
                device=x.device,
                dtype=x.dtype
            )
            # 预测当前时刻的速度场
            v_pred = self._predict_flow(x, t, quantized)
            # 欧拉更新
            x = x + v_pred * dt

        return x
class VQCode2MelCFM_cfm(nn.Module):
    @typechecked
    def __init__(
        self,
        emb_dim: int = 128,
        mel_dim: int = 100,
        fs: int = 24000,
        vq_fr: float = 75.0,
        mel_hop_size: int = 256,
        cfm_sigma: float = 0.0,
        flow_hidden_dim: int = 256,
    ):
        super().__init__()
        # VQCode2Mel components
        self.fs = fs
        self.vq_fr = vq_fr
        self.mel_hop_size = mel_hop_size
        self.emb_dim = emb_dim
        self.mel_dim = mel_dim
        # CFM matcher
        self.matcher = ConditionalFlowMatcher(sigma=cfm_sigma)

        # Flow prediction head\        # input dim: mel_dim + emb_dim + 1 (time)
        self.flow_head = nn.Sequential(
            nn.Linear(mel_dim + emb_dim + 1, flow_hidden_dim),
            nn.ReLU(),
            nn.Linear(flow_hidden_dim, mel_dim),
        )

    def _predict_flow(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor
    ) -> torch.Tensor:
        # cond: [B, T_vq, E] -> resize to [B, T_mel, E]
        if cond.shape[1] != x_t.shape[1]:
            cond = cond.transpose(1,2)                                # [B, E, T_vq]
            cond = F.interpolate(cond, size=x_t.shape[1], mode='linear', align_corners=True)
            cond = cond.transpose(1,2)                                # [B, T_mel, E]
        # time feature
        t_feat = t.view(-1, 1).expand(-1, x_t.shape[1]).unsqueeze(-1)  # [B, T_mel, 1]
        inp = torch.cat([x_t, cond, t_feat], dim=-1)                   # [B, T_mel, mel+E+1]
        return self.flow_head(inp)                                     # [B, T_mel, mel_dim]

    def forward(
        self,
        quantized: torch.Tensor, # [B, codebook_dim, T]
        mel_gt: torch.Tensor # [B, T, mel_dim]
    ) -> torch.Tensor:
        """
        Args:
            vq_code: [B, T_vq] int64  
            mel_gt:  [B, T_mel, mel_dim] float
        Returns:
            mel_pred: [B, T_mel, mel_dim]
            total_loss: scalar tensor (recon + cfm)
        """

        quantized = quantized.transpose(1, 2) #[B, T, codebook_dim]
        mel_gt = mel_gt.transpose(1, 2)
        # 1) 从标准正态采噪声 x0
        x0 = torch.randn_like(mel_gt)

        # 2) CFM 构造 (t, x_t, u_t)
        t, x_t, u_t = self.matcher.sample_location_and_conditional_flow(x0, mel_gt)

        # 5. Predict vector field and compute CFM loss
        v_pred = self._predict_flow(x_t, t, quantized)
        cfm_loss = F.mse_loss(v_pred, u_t)

        return cfm_loss
    @torch.no_grad()
    def decode(
        self,
        quantized: torch.Tensor, # [B, codebook_dim, T]
        num_steps: int = 50
    ) -> torch.Tensor:
        """
        基于训练好的 CFM 流场，对初始 mel_pred 做 num_steps 步欧拉迭代精修。

        Args:
            vq_code: [B, T_vq] Tensor 输入的 VQ code 序列。
            num_steps: int 迭代步数，越大精修越精细，但越慢。
        Returns:
            x: [B, T_mel, mel_dim] FloatTensor，精修后的 Mel 频谱。
        """
        quantized = quantized.transpose(1, 2)
        self.eval()
        target_len = math.floor(self.fs * quantized.shape[1] / self.vq_fr) // self.mel_hop_size

        x = torch.randn(quantized.shape[0], target_len, self.mel_dim, device=quantized.device)

        if num_steps < 2:
            # t = 0 时一次完整修正
            t = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            v_pred = self._predict_flow(x, t, quantized)
            return x + v_pred

        dt = 1.0 / (num_steps - 1)
        # 3) 在 [0,1] 时间区间上做欧拉积分
        for i in range(num_steps):
            t = torch.full(
                (x.size(0),),
                float(i) * dt,
                device=x.device,
                dtype=x.dtype
            )
            # 预测当前时刻的速度场
            v_pred = self._predict_flow(x, t, quantized)
            # 欧拉更新
            x = x + v_pred * dt

        return x
    
    def train_decode(
        self,
        quantized: torch.Tensor, # [B, codebook_dim, T]
        num_steps: int = 50
    ) -> torch.Tensor:
        """
        基于训练好的 CFM 流场，对初始 mel_pred 做 num_steps 步欧拉迭代精修。

        Args:
            vq_code: [B, T_vq] Tensor 输入的 VQ code 序列。
            num_steps: int 迭代步数，越大精修越精细，但越慢。
        Returns:
            x: [B, T_mel, mel_dim] FloatTensor，精修后的 Mel 频谱。
        """
        quantized = quantized.transpose(1, 2)
        target_len = math.floor(self.fs * quantized.shape[1] / self.vq_fr) // self.mel_hop_size

        x = torch.randn(quantized.shape[0], target_len, self.mel_dim, device=quantized.device)

        if num_steps < 2:
            # t = 0 时一次完整修正
            t = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            v_pred = self._predict_flow(x, t, quantized)
            return x + v_pred

        dt = 1.0 / (num_steps - 1)
        # 3) 在 [0,1] 时间区间上做欧拉积分
        for i in range(num_steps):
            t = torch.full(
                (x.size(0),),
                float(i) * dt,
                device=x.device,
                dtype=x.dtype
            )
            # 预测当前时刻的速度场
            v_pred = self._predict_flow(x, t, quantized)
            # 欧拉更新
            x = x + v_pred * dt

        return x

