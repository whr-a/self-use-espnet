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

from espnet2.gan_codec.abs_nongan_codec import AbsNonGANCodec
from espnet2.gan_codec.shared.decoder.seanet import SEANetDecoder
from espnet2.gan_codec.shared.discriminator.msmpmb_discriminator import (
    MultiScaleMultiPeriodMultiBandDiscriminator,
)
from espnet2.gan_codec.shared.encoder.seanet import SEANetEncoder
from espnet2.gan_codec.shared.loss.freq_loss import MultiScaleMelSpectrogramLoss

from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.utils.split_band import split_audio_bands, reconstruct_audio_bands
from espnet2.gan_codec.shared.decoder.cfm import VQCode2MelCFM_cnn_cfm, VQCode2MelCFM_cfm
from espnet2.gan_codec.shared.quantizer.band_vq import BandVectorQuantizer
from meldataset import get_mel_spectrogram
import bigvgan
class Bandcodec_oneenc_vq(AbsNonGANCodec):
    """DAC model."""

    @typechecked
    def __init__(
        self,
        sampling_rate: int = 24000,
        generator_params: Dict[str, Any] = {
            "hidden_dim": 512,
            "codebook_dim": 512,
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
            "cfm_model": "cnn_cfm",
            "quantizer_bins": 1024,
            "quantizer_decay": 0.99,
            "quantizer_kmeans_init": True,
            "quantizer_kmeans_iters": 50,
            "quantizer_threshold_ema_dead_code": 2,
            "quantizer_dropout": False,
            "cfm_mel_dim": 100,
            "cfm_vq_fr": 75.0,
            "cfm_mel_hop_size": 256,
            "cfm_sigma": 0.0,
            "cfm_flow_hidden_dim": 256,
            "cfm_conv_channels": 256,
            "cfm_num_conv_layers": 3,
            "preload": True,
            "fix_encoder": True,
            "preload_path": "/u",
        },
        vocoder_path: str = '/u/hwang41/hwang41/3ai/espnet/espnet2/gan_codec/shared/vocoder/bigvgan',
        vocoder_usecuda: bool = True,
        # loss related
        use_mel_loss: bool = True,
        use_vocoder_mel_loss: bool = True,
        cfm_steps: int = 50,
        mel_loss_params: Dict[str, Any] = {
            "range_start": 6,
            "range_end": 11,
            "window": "hann",
            "n_mels": 80,
            "fmin": 0,
            "fmax": None,
            "log_base": None,
        },
        lambda_quantization: float = 1.0,
        lambda_reconstruct: float = 1.0,
        lambda_mel: float = 45.0,
        lambda_vocoder_mel: float = 45.0,
        lambda_cfm: float = 1.0,
        cache_generator_outputs: bool = False,
    ):
        """Intialize DAC model.

        Args:
             TODO(jiatong)
        """
        super().__init__()

        # define modules
        generator_params.update(
            sample_rate=sampling_rate,
            quantizer_n_bands=len(generator_params["bands"])
        )
        self.bands = generator_params["bands"]
        self.num_bands = len(generator_params["bands"])
        self.vocoder = bigvgan.BigVGAN.from_pretrained(vocoder_path, use_cuda_kernel=vocoder_usecuda)
        self.vocoder.eval()
        for p in self.vocoder.parameters():
            p.requires_grad = False
        self.generator = Bandcodec_oneenc_vqGenerator(**generator_params)
        self.discriminator = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)
        self.use_mel_loss = use_mel_loss
        mel_loss_params.update(fs=sampling_rate)
        self.use_vocoder_mel_loss = use_vocoder_mel_loss
        self.cfm_steps = cfm_steps
        if self.use_vocoder_mel_loss:
            assert self.use_mel_loss, "only use vocoder mel loss with Mel loss"
            self.vocoder_generator_reconstruct_loss = torch.nn.L1Loss(reduction="mean")
            self.vocoder_mel_loss = MultiScaleMelSpectrogramLoss(
                **mel_loss_params,
            )

        # coefficients
        self.lambda_quantization = lambda_quantization
        self.lambda_reconstruct = lambda_reconstruct
        self.lambda_cfm = lambda_cfm

        if self.use_mel_loss:
            self.lambda_mel = lambda_mel
        if self.use_vocoder_mel_loss:
            self.lambda_vocoder_mel = lambda_vocoder_mel

        # cache
        self.cache_generator_outputs = cache_generator_outputs
        self._cache = None

        # store sampling rate for saving wav file
        # (not used for the training)
        self.fs = sampling_rate
        self.num_streams = len(generator_params["bands"])
        self.frame_shift = functools.reduce(
            lambda x, y: x * y, generator_params["encdec_ratios"]
        )
        self.code_size_per_stream = [
            generator_params["quantizer_bins"]
        ] * self.num_streams

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
        return self._forward_generator(
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
        y = audio.squeeze(1)
        subbands = split_audio_bands(y, self.fs, self.bands)  # (B, N_bands, T)
        mel_gt = []
        for i in range(len(self.bands)):
            mel = get_mel_spectrogram(subbands[:,i,:], self.vocoder.h)
            mel_gt.append(mel)
        mel_gt = torch.stack(mel_gt, dim=1)
        mel_len = mel_gt.shape[-1]
        # calculate generator outputs
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False
            audio_hat = (
                self.generator(subbands, mel_gt)
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

        # calculate losses

        vq_loss = audio_hat[0] * self.lambda_quantization
        cfm_loss = audio_hat[1] * self.lambda_cfm

        loss = vq_loss + cfm_loss

        stats = dict(
            vq_loss=vq_loss.item(),
            cfm_loss=cfm_loss.item(),
        )
        
        if self.use_mel_loss:
            subband_mel_loss = 0.0
            quantized = audio_hat[2]
            if self.use_vocoder_mel_loss:
                wav_gen = []
            for i in range(self.num_bands):
                quantized_thisband = quantized[:,i,:,:]
                mel_gt_thisband = mel_gt[:,i,:,:]
                mel_gen = self.generator.decoders[i].train_decode(quantized_thisband, self.cfm_steps)
                mel_gen = mel_gen.transpose(1,2)
                T_gt = mel_gt_thisband.size(-1)
                T_gen = mel_gen.size(-1)
                if T_gen > T_gt:
                    # truncate
                    mel_gen = mel_gen[..., :T_gt]
                elif T_gen < T_gt:
                    # pad with zeros
                    pad = T_gt - T_gen
                    # create zeros of shape (B, mel_dim, pad)
                    zero_pad = mel_gen.new_zeros(mel_gen.size(0), mel_gen.size(1), pad)
                    mel_gen = torch.cat([mel_gen, zero_pad], dim=-1)
                subband_mel_loss += (
                    F.l1_loss(mel_gen, mel_gt_thisband)
                )
                if self.use_vocoder_mel_loss:
                    wav_gen.append(self.vocoder(mel_gen).squeeze(1))
                   
            subband_mel_loss = (subband_mel_loss * self.lambda_mel) / self.num_bands
            loss = loss + subband_mel_loss
            stats.update(subband_mel_loss = subband_mel_loss.item())

            if self.use_vocoder_mel_loss:
                wav_gen = torch.stack(wav_gen, dim=1)
                wav_gen = reconstruct_audio_bands(wav_gen).unsqueeze(1)
                T_gt = audio.size(-1)
                T_gen = wav_gen.size(-1)
                if T_gen > T_gt:
                    # truncate
                    wav_gen = wav_gen[..., :T_gt]
                elif T_gen < T_gt:
                    # pad with zeros
                    pad = T_gt - T_gen
                    # create zeros of shape (B, mel_dim, pad)
                    zero_pad = wav_gen.new_zeros(wav_gen.size(0), wav_gen.size(1), pad)
                    wav_gen = torch.cat([wav_gen, zero_pad], dim=-1)
                vocoder_reconstruct_loss = (
                    self.vocoder_generator_reconstruct_loss(audio, wav_gen) * self.lambda_reconstruct
                )
                vocoder_mel_loss = self.vocoder_mel_loss(wav_gen, audio) * self.lambda_vocoder_mel
                loss = loss + vocoder_reconstruct_loss + vocoder_mel_loss
                stats.update(
                    vocoder_reconstruct_loss=vocoder_reconstruct_loss.item(),
                    vocoder_mel_loss=vocoder_mel_loss.item()
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
        codes = self.generator.encode(x)
        wav = self.decode(codes)

        return {"wav": wav, "codec": codes}

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
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Run encoding.

        Args:
            x (Tensor): Input codes (T_code, N_stream).

        Returns:
            Tensor: Generated waveform (T_wav,).

        """
        quantized = self.generator.decode(x)
        wav_gen = []
        for i in range(self.num_bands):
            quantized_thisband = quantized[:,i,:,:]
            mel_gen = self.generator.decoders[i].decode(quantized_thisband, self.cfm_steps)
            mel_gen = mel_gen.transpose(1,2)
            wav_gen.append(self.vocoder(mel_gen).squeeze(1))
        wav_gen = torch.stack(wav_gen, dim=1)
        wav_gen = reconstruct_audio_bands(wav_gen)
        return wav_gen


class Bandcodec_oneenc_vqGenerator(nn.Module):
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
        hidden_dim: int = 512,
        codebook_dim: int = 512,
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
        cfm_model: str = "cfm",
        quantizer_n_bands: int = 4,
        quantizer_bins: int = 1024,
        quantizer_decay: float = 0.99,
        quantizer_kmeans_init: bool = True,
        quantizer_kmeans_iters: int = 50,
        quantizer_threshold_ema_dead_code: int = 2,
        quantizer_dropout: bool = False, 
        cfm_mel_dim: int = 100,
        cfm_vq_fr: float = 75.0,
        cfm_mel_hop_size: int = 256,
        cfm_sigma: float = 0.0,
        cfm_flow_hidden_dim: int = 256,
        cfm_conv_channels: int = 256,
        cfm_num_conv_layers: int = 3,
        preload: bool = True,
        fix_encoder: bool = True,
        preload_path: str = "/u",
    ):
        """Initialize DAC Generator.

        Args:
            TODO(jiatong)
        """
        super().__init__()

        # Initialize encoder
        self.sample_rate = sample_rate
        self.bands = bands
        self.num_bands = len(bands)
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
        if preload == True:
            ckpt = torch.load(preload_path, map_location="cpu")
            state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
            prefix = "codec.generator.encoder."
            enc_dict = {}
            for k, v in state_dict.items():
                if k.startswith(prefix):
                    new_k = k[len(prefix):]
                    enc_dict[new_k] = v
            incompatible = self.encoder.load_state_dict(enc_dict, strict=False)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    f"Failed to load encoder weights:\n"
                    f"  missing keys: {incompatible.missing_keys}\n"
                    f"  unexpected keys: {incompatible.unexpected_keys}"
                )
        if fix_encoder == True:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.quantizer = BandVectorQuantizer(
            num_bands=len(bands),
            dimension=hidden_dim,
            codebook_dim=codebook_dim,
            bins=quantizer_bins,
            decay=quantizer_decay,
            kmeans_init=quantizer_kmeans_init,
            kmeans_iters=quantizer_kmeans_iters,
            threshold_ema_dead_code=quantizer_threshold_ema_dead_code,
            quantizer_dropout=quantizer_dropout,
        )
        # Initialize decoder
        if cfm_model == "cfm":
            self.decoders = nn.ModuleList([VQCode2MelCFM_cfm(
                    emb_dim=hidden_dim,
                    mel_dim=cfm_mel_dim,
                    fs=sample_rate,
                    vq_fr=cfm_vq_fr,
                    mel_hop_size=cfm_mel_hop_size,
                    cfm_sigma=cfm_sigma,
                    flow_hidden_dim=cfm_flow_hidden_dim
                    # todo: pass parameters, according to bigvgan model
                )
                for _ in bands
            ])
        elif cfm_model == "cnn_cfm":
            self.decoders = nn.ModuleList([VQCode2MelCFM_cnn_cfm(
                    emb_dim=hidden_dim,
                    conv_channels=cfm_conv_channels,
                    num_conv_layers=cfm_num_conv_layers,
                    mel_dim=cfm_mel_dim,
                    vq_fr=cfm_vq_fr,
                    mel_hop_size=cfm_mel_hop_size,
                    cfm_sigma=cfm_sigma,
                    flow_hidden_dim=cfm_flow_hidden_dim
                    # todo: pass parameters, according to bigvgan model
                )
                for _ in bands
            ])

    def forward(self, subbands: torch.Tensor, mel_gt: torch.Tensor):
        """DAC forward propagation.

        Args:
            subbands (torch.Tensor): Input tensor of shape (B, bands, 1, T).
        Returns:
            torch.Tensor: resynthesized audio.
        """

        encoder_out = []
        for i in range(self.num_bands):
            xi = subbands[:, i, :].unsqueeze(1)  # (B,1,T)
            hi = self.encoder(xi)
            encoder_out.append(hi)
        encoder_out = torch.stack(encoder_out, dim=1)

        if self.quantizer.quantizer_dropout == False:
            quantized, codes, bw, vq_loss = self.quantizer(
                encoder_out, self.frame_rate
            )
            cfm_loss_all = []
            for i in range(len(self.bands)):
                quantized_input = quantized[:,i,:,:]
                mel_gt_input = mel_gt[:,i,:,:]
                cfm_loss = self.decoders[i](quantized_input, mel_gt_input)
                cfm_loss_all.append(cfm_loss)
            cfm_loss_all = torch.stack(cfm_loss_all)
            return vq_loss, torch.mean(cfm_loss_all), quantized
        else:
            raise NotImplementedError

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
        subbands = split_audio_bands(x, self.sample_rate, self.bands)
        encoder_out = []
        for i in range(self.num_bands):
            xi = subbands[:, i, :].unsqueeze(1)  # (B,1,T)
            hi = self.encoder(xi)
            encoder_out.append(hi)
        encoder_out = torch.stack(encoder_out, dim=1)

        if self.quantizer.quantizer_dropout == False:
            codes = self.quantizer.encode(
                encoder_out
            )
            return codes
        else:
            raise NotImplementedError

    def decode(self, codes: torch.Tensor):
        """DAC codec decoding.

        Args:
            codecs (torch.Tensor): neural codecs in shape ().
        Returns:
            torch.Tensor: resynthesized audio.
        """
        return self.quantizer.decode(codes)
        


def test_forward_generator_outputs():
    """
    Test the forward pass of the generator for correct keys and types.
    """
    # Instantiate model
    model = Bandcodec_vq()
    model.eval()  # set to eval mode

    # Prepare a dummy audio batch: batch size 2, length equal to frame_shift * 10
    batch_size = 2
    time_steps = model.frame_shift * 10
    audio = torch.randn(batch_size, time_steps)

    # Run forward (generator)
    output = model(audio, forward_generator=True)

    # Check that output is a dict with expected keys
    assert isinstance(output, dict), "Output should be a dict"
    for key in ["loss", "stats", "weight", "optim_idx"]:
        assert key in output, f"Missing key '{key}' in output"

    # Check types
    assert isinstance(output["loss"], torch.Tensor), "Loss should be a Tensor"
    assert isinstance(output["stats"], dict), "Stats should be a dict"
    assert isinstance(output["weight"], int) or isinstance(output["weight"], torch.Tensor), "Weight should be int or Tensor"
    assert isinstance(output["optim_idx"], int), "optim_idx should be an int"


def test_forward_skip_generator():
    """
    Test the forward pass when skipping the generator (optim_idx=1).
    """
    model = Bandcodec_vq()
    model.eval()

    batch_size = 1
    time_steps = model.frame_shift * 5
    audio = torch.randn(batch_size, time_steps)

    # Skip generator forward
    output = model(audio, forward_generator=False)
    assert output["loss"] == 0.0, "Loss should be zero when generator is skipped"
    assert output["optim_idx"] == 1, "optim_idx should be 1 when skipping generator"


def test_inference_shape_and_types():
    """
    Test the inference method for output shapes and types.
    """
    model = Bandcodec_vq()
    model.eval()

    # Single-channel waveform for inference
    time_steps = model.frame_shift * 6
    x = torch.randn(1, time_steps)

    out = model.inference(x)
    assert isinstance(out, dict), "Inference output should be a dict"
    assert "wav" in out and "codec" in out, "Inference output missing keys"

    wav = out["wav"]
    codec = out["codec"]

    # Check tensor types
    assert isinstance(wav, torch.Tensor), "wav should be a Tensor"
    assert isinstance(codec, torch.Tensor) or isinstance(codec, list), "codec should be Tensor or list"

    # Check waveform length matches input length
    assert wav.shape[-1] == x.shape[-1], "Output waveform length should match input"


def test_encode_decode_roundtrip():
    """
    Test that encoding then decoding returns a waveform of expected shape.
    """
    model = Bandcodec_vq()
    model.eval()

    time_steps = model.frame_shift * 8
    x = torch.randn(1, time_steps)

    # Encode
    codes = model.encode(x)
    assert isinstance(codes, torch.Tensor), "Encoded output should be a list of tensors"

    # Decode
    y = model.decode(codes)
    assert isinstance(y, torch.Tensor), "Decoded output should be a Tensor"
    # Should be shape (1, 1, T)
    assert y.dim() == 2 and y.shape[-1] == x.shape[-1], f"Decoded output should have shape (1,1,{x.shape[-1]})"
if __name__ == "__main__":
    test_forward_generator_outputs()
