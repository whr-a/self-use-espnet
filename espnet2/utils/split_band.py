
from typing import List, Tuple

import torch
import torchaudio
import warnings

def split_audio_bands(
    y: torch.Tensor,
    sr: int,
    bands: List[Tuple[float, float]],
    n_fft: int = 2048,
    hop_length: int = None,
    win_length: int = None,
    window_type: str = "hann"
) -> torch.Tensor:
    """
    Split a batch of audio signals into subbands via STFT masking.

    Args:
        y (Tensor): Input tensor of shape (B, T) or (T,).
        sr (int): Sampling rate.
        bands (List[Tuple[float, float]]): List of (low_freq, high_freq) in Hz.
        n_fft (int): FFT size.
        hop_length (int, optional): Hop length between frames. Defaults to n_fft // 2.
        win_length (int, optional): Window length. Defaults to n_fft.
        window_type (str): Window type, either "hann" or "hamming".

    Returns:
        Tensor: Subbands of shape (B, N_bands, T), where N_bands = len(bands).
    """
    # ensure batch dimension
    if y.dim() == 1:
        y = y.unsqueeze(0)  # (1, T)
    B, T = y.shape

    hop_length = hop_length or (n_fft // 2)
    win_length = win_length or n_fft

    # select window
    if window_type == "hann":
        window = torch.hann_window(win_length, periodic=False, device=y.device)
    elif window_type == "hamming":
        window = torch.hamming_window(win_length, periodic=False, device=y.device)
    else:
        raise ValueError(f"Unsupported window type: {window_type}")

    # batched STFT -> (B, F, T_frames)
    S = torch.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
        center=True
    )

    # frequency bins (F,)
    F_bins = S.size(1)
    freqs = torch.linspace(0, sr / 2, F_bins, device=y.device)

    # mask per band and ISTFT
    subbands = []
    for low_f, high_f in bands:
        mask = ((freqs >= low_f) & (freqs < high_f)).view(1, F_bins, 1)
        Sk = S * mask  # (B, F, T_frames)
        try:
            yk = torch.istft(
                Sk,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                length=T,
                center=True
            )  # (B, T)
        except RuntimeError as err:
            if 'window overlap add min' in str(err):
                try:
                    warnings.warn(
                        "ISTFT NOLA check failed for hann window; "
                        "falling back to hamming window for this band.",
                        UserWarning
                    )
                    fallback_win1 = torch.hamming_window(
                        win_length, periodic=False, device=y.device
                    )
                    yk = torch.istft(
                        Sk,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        win_length=win_length,
                        window=fallback_win1,
                        length=T,
                        center=True,
                    )
                except RuntimeError as err2:
                    warnings.warn(
                        "ISTFT NOLA check failed for hamming window; "
                        "falling back to Bartlett window for this band.",
                        UserWarning
                    )
                    if 'window overlap add min' in str(err2):
                        fallback_win2 = torch.bartlett_window(
                            win_length, periodic=False, device=y.device
                        )
                        yk = torch.istft(
                            Sk,
                            n_fft=n_fft,
                            hop_length=hop_length,
                            win_length=win_length,
                            window=fallback_win2,
                            length=T,
                            center=True,
                        )
                    else:
                        raise    
            else:
                raise
        subbands.append(yk)

    # stack into (B, N_bands, T)
    return torch.stack(subbands, dim=1)


def reconstruct_audio_bands(
    subbands: torch.Tensor
) -> torch.Tensor:
    """
    Reconstruct audio by summing subbands.

    Args:
        subbands (Tensor): Tensor of shape (B, N_bands, T).

    Returns:
        Tensor: Reconstructed audio of shape (B, T) or (T,).
    """
    y_recon = subbands.sum(dim=1)  # sum over bands -> (B, T)
    return y_recon


if __name__ == "__main__":
    import os
    import time
    # Example usage and timing
    wav_path = "test/gt/speech_test.wav"
    y, sr = torchaudio.load(wav_path)    # y: (1, T)
    y = y.squeeze(0)                     # -> (T,)

    # Create a batch of 16 identical signals
    batch_size = 16
    y = y.repeat(batch_size, 1)          # -> (16, T)

    # Define subbands in Hz
    bands = [
        (0, 3600),
        (3600, 8000),
        (8000, 12000),
        (12000, 24000)
    ]

    start = time.time()
    subbands, y_recon = split_and_reconstruct(y, sr, bands)
    elapsed = time.time() - start
    print(f"Batched split & reconstruct (batch={batch_size}): {elapsed:.4f} sec")

    # Save one example to verify
    os.makedirs("test/output", exist_ok=True)
    torchaudio.save(
        "test/output/reconstructed.wav",
        y_recon[0].unsqueeze(0),
        sample_rate=sr
    )
