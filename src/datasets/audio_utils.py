from typing import List, Tuple

import torch
import torchaudio


def apply_preprocessing(
        waveform: torch.Tensor,
        audio_sample_rate: int,
        **kwargs
) -> Tuple[torch.Tensor, int]:
    """
    Apply preprocessing steps to the input waveform.
    """
    waveform = waveform.to(torch.float32)
    target_sample_rate = kwargs.get('sample_rate', -1)

    if target_sample_rate != -1 and audio_sample_rate != target_sample_rate:
        waveform, audio_sample_rate = resample_wave(waveform, audio_sample_rate, target_sample_rate)

    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform[:1]

    # Trim too long utterances and apply other sox effects
    if kwargs.get('use_sox_effects', True):
        sox_effects = kwargs.get('sox_effects', [])
        waveform, audio_sample_rate = apply_sox_effects(waveform, audio_sample_rate, sox_effects)

    # Pad too short utterances
    if kwargs.get('padding', True):
        waveform = apply_pad(waveform, kwargs.get('n_frames', 64_600))

    return waveform, audio_sample_rate


def resample_wave(
        waveform: torch.Tensor,
        sample_rate: int,
        target_sample_rate: int
) -> Tuple[torch.Tensor, int]:
    """
    Resample the input waveform to the target sample rate.
    """

    effects = [["rate", str(target_sample_rate)]]

    return torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, effects)  # returns waveform, sample_rate


def resample_file(
        path: str,
        target_sample_rate: int,
        normalize: bool = True
) -> Tuple[torch.Tensor, int]:
    """
    Resample an audio file to the target sample rate.
    """
    effects = [["rate", str(target_sample_rate)]]
    return torchaudio.sox_effects.apply_effects_file(path, effects, normalize=normalize)


def apply_sox_effects(
        waveform: torch.Tensor,
        sample_rate: int,
        effects: List[List[str]]
) -> Tuple[torch.Tensor, int]:
    """Apply a chain of sox effects to the waveform."""

    new_waveform, new_sample_rate = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, effects)
    if new_waveform.shape[1] > 0:
        return new_waveform, new_sample_rate
    return waveform, sample_rate


def apply_pad(waveform: torch.Tensor, cut: int) -> torch.Tensor:
    """
    Pad the waveform by repeating the signal until the desired length is achieved.
    """
    waveform = waveform.squeeze(0)
    waveform_len = waveform.shape[0]

    if waveform_len >= cut:
        return waveform[:cut]

    num_repeats = (cut - 1) // waveform_len + 1
    return torch.tile(waveform, (1, num_repeats))[:, :cut].squeeze(0)
