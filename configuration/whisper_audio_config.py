from pydantic import BaseModel, Field


class _WhisperAudioConfig(BaseModel):
    n_fft: int = Field(default=400, description="Number of FFT points")
    n_mels: int = Field(default=80, description="Number of Mel filterbanks")
    sample_rate: int = Field(default=16_000, description="Target sampling rate in Hz")
    hop_length: int = Field(default=160, description="Hop length for STFT")
    chunk_length: int = Field(default=4, description="Length of the chunk")

    n_samples: int = Field(default=64_600, description="Number of samples in the chunk")
    n_frames: int = Field(default=64_600, description="Number of frames number in a mel spectrogram input")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


if __name__ == '__main__':
    print(_WhisperAudioConfig().model_dump())
