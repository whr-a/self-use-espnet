# Copyright 2024 Yihan Wu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""DAC Modules."""
import functools
import math
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from typeguard import typechecked

from espnet2.gan_codec.abs_gan_codec import AbsGANCodec
from espnet2.gan_codec.shared.decoder.seanet import SEANetDecoder
from espnet2.gan_codec.shared.discriminator.msmpmb_discriminator import (
    MultiScaleMultiPeriodMultiBandDiscriminator,
)
from espnet2.gan_codec.shared.encoder.seanet import SEANetEncoder
from espnet2.gan_codec.shared.loss.freq_loss import MultiScaleMelSpectrogramLoss

from espnet2.gan_tts.hifigan.loss import (
    DiscriminatorAdversarialLoss,
    FeatureMatchLoss,
    GeneratorAdversarialLoss,
)
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.utils.split_band import split_audio_bands, reconstruct_audio_bands

class Bandcodec_oneencdec(AbsGANCodec):
    """DAC model."""

    @typechecked
    def __init__(
        self,
        sampling_rate: int = 24000,
        generator_params: Dict[str, Any] = {
            "hidden_dim": 128,
            "bands": [
                    (0, 3600),
                    (3600, 8000),
                    (8000, 16000),
                    (16000, 24000),
                ],
            "encdec_channels": 1,
            "encdec_n_filters": 32,
            "encdec_n_residual_layers": 1,
            "encdec_ratios": [8, 5, 4, 2],
            "encdec_activation": "Snake",
            "encdec_activation_params": {},
            "encdec_norm": "weight_norm",
            "encdec_norm_params": {},
            "encdec_kernel_size": 7,
            "encdec_residual_kernel_size": 7,
            "encdec_last_kernel_size": 7,
            "encdec_dilation_base": 2,
            "encdec_causal": False,
            "encdec_pad_mode": "reflect",
            "encdec_true_skip": False,
            "encdec_compress": 2,
            "encdec_lstm": 2,
            "decoder_trim_right_ratio": 1.0,
            "decoder_final_activation": None,
            "decoder_final_activation_params": None,
        },
        discriminator_params: Dict[str, Any] = {
            "scale_follow_official_norm": False,
            "msmpmb_discriminator_params": {
                "rates": [],
                "fft_sizes": [2048, 1024, 512],
                "sample_rate": 24000,
                "periods": [2, 3, 5, 7, 11],
                "period_discriminator_params": {
                    "in_channels": 1,
                    "out_channels": 1,
                    "kernel_sizes": [5, 3],
                    "channels": 32,
                    "downsample_scales": [3, 3, 3, 3, 1],
                    "max_downsample_channels": 1024,
                    "bias": True,
                    "nonlinear_activation": "LeakyReLU",
                    "nonlinear_activation_params": {"negative_slope": 0.1},
                    "use_weight_norm": True,
                    "use_spectral_norm": False,
                },
                "band_discriminator_params": {
                    "hop_factor": 0.25,
                    "sample_rate": 24000,
                    "bands": [
                        (0.0, 0.1),
                        (0.1, 0.25),
                        (0.25, 0.5),
                        (0.5, 0.75),
                        (0.75, 1.0),
                    ],
                    "channel": 32,
                },
            },
        },
        # loss related
        generator_adv_loss_params: Dict[str, Any] = {
            "average_by_discriminators": False,
            "loss_type": "mse",
        },
        discriminator_adv_loss_params: Dict[str, Any] = {
            "average_by_discriminators": False,
            "loss_type": "mse",
        },
        use_feat_match_loss: bool = True,
        feat_match_loss_params: Dict[str, Any] = {
            "average_by_discriminators": False,
            "average_by_layers": False,
            "include_final_outputs": True,
        },
        use_mel_loss: bool = True,
        mel_loss_params: Dict[str, Any] = {
            "fs": 24000,
            "range_start": 6,
            "range_end": 11,
            "window": "hann",
            "n_mels": 80,
            "fmin": 0,
            "fmax": None,
            "log_base": None,
        },
        use_dual_decoder: bool = True,
        lambda_reconstruct: float = 1.0,
        lambda_adv: float = 1.0,
        lambda_feat_match: float = 2.0,
        lambda_mel: float = 45.0,
        cache_generator_outputs: bool = False,
    ):
        """Intialize DAC model.

        Args:
             TODO(jiatong)
        """
        super().__init__()
        self.num_bands = len(generator_params["bands"])
        # define modules
        generator_params.update(sample_rate=sampling_rate)
        self.generator = Bandcodec_oneencdecGenerator(**generator_params)
        self.discriminator = Bandcodec_oneencdecDiscriminator(**discriminator_params)
        self.generator_adv_loss = GeneratorAdversarialLoss(
            **generator_adv_loss_params,
        )
        self.generator_reconstruct_loss = torch.nn.L1Loss(reduction="mean")
        self.discriminator_adv_loss = DiscriminatorAdversarialLoss(
            **discriminator_adv_loss_params,
        )
        self.use_feat_match_loss = use_feat_match_loss
        if self.use_feat_match_loss:
            self.feat_match_loss = FeatureMatchLoss(
                **feat_match_loss_params,
            )
        self.use_mel_loss = use_mel_loss
        mel_loss_params.update(fs=sampling_rate)
        if self.use_mel_loss:
            self.mel_loss = MultiScaleMelSpectrogramLoss(
                **mel_loss_params,
            )
        self.use_dual_decoder = use_dual_decoder
        if self.use_dual_decoder:
            assert self.use_mel_loss, "only use dual decoder with Mel loss"

        # coefficients
        self.lambda_reconstruct = lambda_reconstruct
        self.lambda_adv = lambda_adv
        if self.use_feat_match_loss:
            self.lambda_feat_match = lambda_feat_match
        if self.use_mel_loss:
            self.lambda_mel = lambda_mel

        # cache
        self.cache_generator_outputs = cache_generator_outputs
        self._cache = None

        # store sampling rate for saving wav file
        # (not used for the training)
        self.fs = sampling_rate
        self.num_streams = 1
        self.frame_shift = functools.reduce(
            lambda x, y: x * y, generator_params["encdec_ratios"]
        )
        self.code_size_per_stream = None

    def meta_info(self) -> Dict[str, Any]:
        return {
            "fs": self.fs,
            "num_streams": self.num_streams,
            "frame_shift": self.frame_shift,
            "code_size_per_stream": self.code_size_per_stream,
        }

    def forward(
        self,
        audio: torch.Tensor,
        forward_generator: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform generator forward.

        Args:
            audio (Tensor): Audio waveform tensor (B, T_wav).
            forward_generator (bool): Whether to forward generator.

        Returns:
            Dict[str, Any]:
                - loss (Tensor): Loss scalar tensor.
                - stats (Dict[str, float]): Statistics to be monitored.
                - weight (Tensor): Weight tensor to summarize losses.
                - optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        if forward_generator:
            return self._forward_generator(
                audio=audio,
                **kwargs,
            )
        else:
            return self._forward_discrminator(
                audio=audio,
                **kwargs,
            )

    def _forward_generator(
        self,
        audio: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform generator forward.

        Args:
            audio (Tensor): Audio waveform tensor (B, T_wav).

        Returns:
            Dict[str, Any]:
                - loss (Tensor): Loss scalar tensor.
                - stats (Dict[str, float]): Statistics to be monitored.
                - weight (Tensor): Weight tensor to summarize losses.
                - optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        # setup
        batch_size = audio.size(0)

        # TODO(jiatong): double check the multi-channel input
        audio = audio.unsqueeze(1)

        # calculate generator outputs
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False
            audio_hat = (
                self.generator(audio)
            )
        else:
            audio_hat = (
                self._cache
            )

        # store cache
        if self.training and self.cache_generator_outputs and not reuse_cache:
            self._cache = (
                audio_hat
            )

        # calculate discriminator outputs
        p_hat = self.discriminator(audio_hat[0])
        with torch.no_grad():
            # do not store discriminator gradient in generator turn
            p = self.discriminator(audio)

        # calculate losses
        adv_loss = self.generator_adv_loss(p_hat)
        adv_loss = adv_loss * self.lambda_adv
        subband_reconstruct_loss = 0.0
        for i in range(self.num_bands):
            subband_reconstruct_loss += (
                self.generator_reconstruct_loss(audio_hat[1][:, i, :].unsqueeze(1), audio_hat[2][:, i, :].unsqueeze(1))
            )
        subband_reconstruct_loss = (subband_reconstruct_loss * self.lambda_reconstruct) / self.num_bands
        reconstruct_loss = (
            self.generator_reconstruct_loss(audio, audio_hat[0]) * self.lambda_reconstruct
        )
        loss = adv_loss + subband_reconstruct_loss + reconstruct_loss
        stats = dict(
            adv_loss=adv_loss.item(),
            subband_reconstruct_loss = subband_reconstruct_loss.item(),
            reconstruct_loss=reconstruct_loss.item(),
        )
        if self.use_feat_match_loss:
            feat_match_loss = self.feat_match_loss(p_hat, p)
            feat_match_loss = feat_match_loss * self.lambda_feat_match
            loss = loss + feat_match_loss
            stats.update(feat_match_loss=feat_match_loss.item())
        if self.use_mel_loss:
            subband_mel_loss = 0.0
            for i in range(self.num_bands):
                subband_mel_loss += (
                    self.mel_loss(audio_hat[2][:, i, :].unsqueeze(1), audio_hat[1][:, i, :].unsqueeze(1))
                )
            subband_mel_loss = (subband_mel_loss * self.lambda_mel) / self.num_bands
            mel_loss = self.mel_loss(audio_hat[0], audio)
            mel_loss = self.lambda_mel * mel_loss
            loss = loss + subband_mel_loss + mel_loss
            stats.update(
                mel_loss=mel_loss.item(),
                subband_mel_loss = subband_mel_loss.item()
            )

        stats.update(loss=loss.item())

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        # reset cache
        if reuse_cache or not self.training:
            self._cache = None

        return {
            "loss": loss,
            "stats": stats,
            "weight": weight,
            "optim_idx": 0,  # needed for trainer
        }

    def _forward_discrminator(
        self,
        audio: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform generator forward.

        Args:
            audio (Tensor): Audio waveform tensor (B, T_wav).

        Returns:
            Dict[str, Any]:
                - loss (Tensor): Loss scalar tensor.
                - stats (Dict[str, float]): Statistics to be monitored.
                - weight (Tensor): Weight tensor to summarize losses.
                - optim_idx (int): Optimizer index (0 for G and 1 for D).

        """

        # setup
        batch_size = audio.size(0)
        audio = audio.unsqueeze(1)

        # calculate generator outputs
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False
            audio_hat= (
                self.generator(
                    audio
                )
            )
        else:
            audio_hat= (
                self._cache
            )

        # store cache
        if self.cache_generator_outputs and not reuse_cache:
            self._cache = (
                audio_hat,
            )

        # calculate discriminator outputs
        p_hat = self.discriminator(audio_hat[0].detach())
        p = self.discriminator(audio)

        # calculate losses
        real_loss, fake_loss = self.discriminator_adv_loss(p_hat, p)
        loss = real_loss + fake_loss

        stats = dict(
            discriminator_loss=loss.item(),
            real_loss=real_loss.item(),
            fake_loss=fake_loss.item(),
        )
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        # reset cache
        if reuse_cache or not self.training:
            self._cache = None

        return {
            "loss": loss,
            "stats": stats,
            "weight": weight,
            "optim_idx": 1,  # needed for trainer
        }

    def inference(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Run inference.

        Args:
            x (Tensor): Input audio (T_wav,).

        Returns:
            Dict[str, Tensor]:
                * wav (Tensor): Generated waveform tensor (T_wav,).
                * codec (Tensor): Generated neural codec (T_code, N_stream).

        """
        codec = self.generator.encode(x)
        wav = self.generator.decode(codec)

        return {"wav": wav, "codec": codec}

    def encode(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Run encoding.

        Args:
            x (Tensor): Input audio (T_wav,).

        Returns:
            Tensor: Generated codes (T_code, N_stream).

        """
        return self.generator.encode(x)

    def decode(
        self,
        x: List[torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """Run encoding.

        Args:
            x (Tensor): Input codes (T_code, N_stream).

        Returns:
            Tensor: Generated waveform (T_wav,).

        """
        return self.generator.decode(x)


class Bandcodec_oneencdecGenerator(nn.Module):
    """DAC generator module."""

    @typechecked
    def __init__(
        self,
        sample_rate: int = 24000,
        bands: List[Any] = [
                    (0, 3600),
                    (3600, 8000),
                    (8000, 16000),
                    (16000, 24000),
                ],
        hidden_dim: int = 128,
        encdec_channels: int = 1,
        encdec_n_filters: int = 32,
        encdec_n_residual_layers: int = 1,
        encdec_ratios: List[int] = [8, 5, 4, 2],
        encdec_activation: str = "Snake",
        encdec_activation_params: Dict[str, Any] = {},
        encdec_norm: str = "weight_norm",
        encdec_norm_params: Dict[str, Any] = {},
        encdec_kernel_size: int = 7,
        encdec_residual_kernel_size: int = 7,
        encdec_last_kernel_size: int = 7,
        encdec_dilation_base: int = 2,
        encdec_causal: bool = False,
        encdec_pad_mode: str = "reflect",
        encdec_true_skip: bool = False,
        encdec_compress: int = 2,
        encdec_lstm: int = 2,
        decoder_trim_right_ratio: float = 1.0,
        decoder_final_activation: Optional[str] = None,
        decoder_final_activation_params: Optional[dict] = None,
    ):
        """Initialize DAC Generator.

        Args:
            TODO(jiatong)
        """
        super().__init__()

        # Initialize encoder
        self.sample_rate = sample_rate
        self.bands = bands
        self.frame_rate = math.ceil(sample_rate / np.prod(encdec_ratios))

        self.encoder = SEANetEncoder(
                channels=encdec_channels,
                dimension=hidden_dim,
                n_filters=encdec_n_filters,
                n_residual_layers=encdec_n_residual_layers,
                ratios=encdec_ratios,
                activation=encdec_activation,
                activation_params=encdec_activation_params,
                norm=encdec_norm,
                norm_params=encdec_norm_params,
                kernel_size=encdec_kernel_size,
                residual_kernel_size=encdec_residual_kernel_size,
                last_kernel_size=encdec_last_kernel_size,
                dilation_base=encdec_dilation_base,
                causal=encdec_causal,
                pad_mode=encdec_pad_mode,
                true_skip=encdec_true_skip,
                compress=encdec_compress,
                lstm=encdec_lstm,
            )

        # Initialize decoder
        self.decoder = SEANetDecoder(
                channels=encdec_channels,
                dimension=hidden_dim,
                n_filters=encdec_n_filters,
                n_residual_layers=encdec_n_residual_layers,
                ratios=encdec_ratios,
                activation=encdec_activation,
                activation_params=encdec_activation_params,
                norm=encdec_norm,
                norm_params=encdec_norm_params,
                kernel_size=encdec_kernel_size,
                residual_kernel_size=encdec_residual_kernel_size,
                last_kernel_size=encdec_last_kernel_size,
                dilation_base=encdec_dilation_base,
                causal=encdec_causal,
                pad_mode=encdec_pad_mode,
                true_skip=encdec_true_skip,
                compress=encdec_compress,
                lstm=encdec_lstm,
                trim_right_ratio=decoder_trim_right_ratio,
                final_activation=decoder_final_activation,
                final_activation_params=decoder_final_activation_params,
            )

    def forward(self, x: torch.Tensor, use_dual_decoder: bool = False):
        """DAC forward propagation.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, T).
        Returns:
            torch.Tensor: resynthesized audio.
        """
        y = x.squeeze(1) # (B, T)
        subbands = split_audio_bands(y, self.sample_rate, self.bands)  # (B, N_bands, T)

        outputs = []
        for i in range(len(self.bands)):
            xi = subbands[:, i, :].unsqueeze(1)  # (B,1,T)
            hi = self.encoder(xi)
            yi = self.decoder(hi)           # (B,1,T)
            outputs.append(yi.squeeze(1))

        stacked = torch.stack(outputs, dim=1) # (B, N_bands, T)
        y_rec = reconstruct_audio_bands(stacked) # (B, T)

        return y_rec.unsqueeze(1), subbands, stacked

    def encode(
        self,
        x: torch.Tensor,
    ):
        """DAC codec encoding.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, T).
        Returns:
            torch.Tensor: neural codecs in shape ().
        """

        y = x.squeeze(1)  # (B, T)
        subbands = split_audio_bands(y, self.sample_rate, self.bands)  # (B, N_bands, T)

        # Encode each band separately
        hidden_states: List[torch.Tensor] = []
        for i in range(len(self.bands)):
            xi = subbands[:, i, :].unsqueeze(1)  # (B,1,T)
            hi = self.encoder(xi)
            hidden_states.append(hi)
        return hidden_states

    def decode(self, hidden_states: List[torch.Tensor]):
        """DAC codec decoding.

        Args:
            codecs (torch.Tensor): neural codecs in shape ().
        Returns:
            torch.Tensor: resynthesized audio.
        """
        outputs: List[torch.Tensor] = []
        for i, hi in enumerate(hidden_states):
            yi = self.decoder(hi)  # (B, 1, T)
            outputs.append(yi.squeeze(1))

        # Stack and sum across bands to reconstruct
        stacked = torch.stack(outputs, dim=1)         # (B, N_bands, T)
        y_rec = reconstruct_audio_bands(stacked)     # (B, T)
        return y_rec.unsqueeze(1)                    # (B, 1, T)


class Bandcodec_oneencdecDiscriminator(nn.Module):
    """DAC discriminator module."""

    def __init__(
        self,
        # MultiScaleMultiPeriodMultiBandDiscriminator parameters
        msmpmb_discriminator_params: Dict[str, Any] = {
            "rates": [],
            "fft_sizes": [2048, 1024, 512],
            "sample_rate": 24000,
            "periods": [2, 3, 5, 7, 11],
            "period_discriminator_params": {
                "in_channels": 1,
                "out_channels": 1,
                "kernel_sizes": [5, 3],
                "channels": 32,
                "downsample_scales": [3, 3, 3, 3, 1],
                "max_downsample_channels": 1024,
                "bias": True,
                "nonlinear_activation": "LeakyReLU",
                "nonlinear_activation_params": {"negative_slope": 0.1},
                "use_weight_norm": True,
                "use_spectral_norm": False,
            },
            "band_discriminator_params": {
                "hop_factor": 0.25,
                "sample_rate": 24000,
                "bands": [
                    (0.0, 0.1),
                    (0.1, 0.25),
                    (0.25, 0.5),
                    (0.5, 0.75),
                    (0.75, 1.0),
                ],
                "channel": 32,
            },
        },
        scale_follow_official_norm: bool = False,
    ):
        """Initialize DAC Discriminator module.

        Args:

        """
        super().__init__()

        self.msmpmb_discriminator = MultiScaleMultiPeriodMultiBandDiscriminator(
            **msmpmb_discriminator_params
        )

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List[List[Tensor]]: List of list of each discriminator outputs,
                which consists of each layer output tensors. Multi scale and
                multi period ones are concatenated.

        """
        msmpmb_outs = self.msmpmb_discriminator(x)
        return msmpmb_outs
