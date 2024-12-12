from typing import List, Callable, Literal

import torch
import torchaudio

from src import model_configuration


def create_mfcc_transform(**kwargs) -> torchaudio.transforms.MFCC:
    """Create an MFCC transform based on the given configuration."""
    return torchaudio.transforms.MFCC(
        sample_rate=kwargs.get("sample_rate", model_configuration.audio_config.sample_rate),
        n_mfcc=kwargs.get("n_mfcc", model_configuration.audio_config.n_mfcc),
        melkwargs={
            "n_fft": kwargs.get("n_fft", model_configuration.audio_config.n_fft),
            "win_length": kwargs.get("win_length", model_configuration.audio_config.win_length),
            "hop_length": kwargs.get("hop_length", model_configuration.audio_config.hop_length),
        },
    ).to(model_configuration.device)


def create_lfcc_transform(**kwargs) -> torchaudio.transforms.LFCC:
    """Create an LFCC transform based on the given configuration."""
    return torchaudio.transforms.LFCC(
        sample_rate=kwargs.get("sample_rate", model_configuration.audio_config.sample_rate),
        n_lfcc=kwargs.get("n_lfcc", model_configuration.audio_config.n_lfcc),
        speckwargs={
            "n_fft": kwargs.get("n_fft", model_configuration.audio_config.n_fft),
            "win_length": kwargs.get("win_length", model_configuration.audio_config.win_length),
            "hop_length": kwargs.get("hop_length", model_configuration.audio_config.hop_length),
        },
    ).to(model_configuration.device)


def create_mel_scale_transform(**kwargs) -> torchaudio.transforms.MelScale:
    """Create a MelScale transform based on the given configuration."""
    return torchaudio.transforms.MelScale(
        n_mels=kwargs.get("n_mels", model_configuration.audio_config.n_mels),
        n_stft=kwargs.get("n_stft", model_configuration.audio_config.n_stft),
        sample_rate=kwargs.get("sample_rate", model_configuration.audio_config.sample_rate),
    ).to(model_configuration.device)


def create_delta_transform(**kwargs) -> torchaudio.transforms.ComputeDeltas:
    """Create a ComputeDeltas transform based on the given configuration."""
    return torchaudio.transforms.ComputeDeltas(
        win_length=kwargs.get("delta_win_length", model_configuration.audio_config.delta_win_length),
        mode="replicate",
    )


def prepare_feature(
        audio: torch.Tensor,
        feature_type: Literal["mfcc", "lfcc"],
        **kwargs
) -> torch.Tensor:
    """
    Prepare the given audio using either MFCC or LFCC.
    """

    if audio.ndim < 4:
        audio = audio.unsqueeze(1)  # (bs, 1, n_lfcc/n_mfcc, frames)

    transform_fn = create_mfcc_transform(**kwargs) if feature_type == "mfcc" else create_lfcc_transform(**kwargs)

    x = transform_fn(audio)

    return x[:, :, :, :kwargs.get("max_frames", model_configuration.audio_config.max_frames)]


def prepare_double_delta(
        audio: torch.Tensor,
        feature_type: Literal["mfcc", "lfcc"],
        **kwargs
) -> torch.Tensor:
    """
    Prepare double delta features for the given audio using either MFCC or LFCC.
    """

    if audio.ndim < 4:
        audio = audio.unsqueeze(1)  # (bs, 1, n_lfcc/n_mfcc, frames)

    transform_fn = create_mfcc_transform(**kwargs) if feature_type == "mfcc" else create_lfcc_transform(**kwargs)

    x = transform_fn(audio)
    delta_fn = create_delta_transform(**kwargs)
    delta = delta_fn(x)
    double_delta = delta_fn(delta)
    x = torch.cat((x, delta, double_delta), dim=2)  # -> [bs, 1, 128 * 3, 1500]

    return x[:, :, :, :kwargs.get("max_frames", model_configuration.audio_config.max_frames)]


def prepare_mfcc_double_delta(audio: torch.Tensor, **kwargs) -> torch.Tensor:
    """Prepare MFCC double delta features for the given audio."""
    return prepare_double_delta(audio, feature_type="mfcc", **kwargs)


def prepare_lfcc(audio: torch.Tensor, **kwargs) -> torch.Tensor:
    """Prepare LFCC features for the given audio."""
    return prepare_double_delta(audio, feature_type="lfcc", **kwargs)


def prepare_mfcc(audio: torch.Tensor, **kwargs) -> torch.Tensor:
    """Prepare MFCC features for the given audio."""
    return prepare_double_delta(audio, feature_type="mfcc", **kwargs)


def prepare_lfcc_double_delta(audio: torch.Tensor, **kwargs) -> torch.Tensor:
    """Prepare MFCC double delta features for the given audio."""
    return prepare_double_delta(audio, feature_type="lfcc", **kwargs)


def get_frontend(
        frontends: List[Literal["mfcc", "lfcc", "mfcc_double_delta", "lfcc_double_delta"]]
) -> Callable:
    """
    Get the appropriate frontend function based on the given list of frontends.
    """

    if "mfcc" in frontends:
        # return prepare_mfcc
        return prepare_mfcc_double_delta
    elif "lfcc" in frontends:
        # return prepare_lfcc
        return prepare_lfcc_double_delta
    elif "mfcc_double_delta" in frontends:
        return prepare_mfcc_double_delta
    elif "lfcc_double_delta" in frontends:
        return prepare_lfcc_double_delta
    raise ValueError(f"{frontends} frontend is not supported!")
