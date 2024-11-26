from typing import List, Dict, Union

from pydantic import BaseModel, Field, field_serializer


class _AudioConfig(BaseModel):
    # audio processing parameters
    sample_rate: int = Field(default=16_000, description="Target sampling rate in Hz")
    win_length: int = Field(default=400, description="Window length for STFT")
    hop_length: int = Field(default=160, description="Hop length for STFT")
    chunk_length: int = Field(default=4, description="Length of the chunk")
    n_samples: int = Field(default=None, description="Number of samples in the chunk")
    n_mfcc: int = Field(default=128, description="Number of MFCC coefficients")
    n_lfcc: int = Field(default=128, description="Number of LFCC coefficients")

    n_fft: int = Field(default=512, description="Number of FFT points")
    n_mels: int = Field(default=80, description="Number of Mel filterbanks")
    n_stft: int = Field(default=257, description="Number of bins in STFT")
    delta_win_length: int = Field(default=400, description="Window length for computing delta")
    max_frames: int = Field(default=3000, description="Maximum number of frames to keep")
    n_frames: int = Field(default=64_600, description="Number of frames number in a mel spectrogram input")
    number_of_coefficients: int = Field(default=384, description="Number of coefficients to keep")

    normalize: bool = Field(default=True, description="Whether to normalize audio")
    padding: bool = Field(default=True, description="Whether to apply padding")
    frames_number: int = Field(default=400, description="Number of frames for padding")
    use_sox_effects: bool = Field(default=True, description="Whether to apply sox effects (by default True -> trimming "
                                                            "the silence")
    sox_effects: List[Dict[str, Union[int, float, str]]] = Field(
        default=[
            {
                "effect": "silence",
                "mode": 1,
                "beginning_duration": 0.2,
                "beginning_threshold": "1%",
                "middle_end_indicator": -1,
                "middle_end_duration": 0.2,
                "middle_end_threshold": "1%"
            },
            # Example of other effects:
            # {
            #     "effect": "reverb",
            #     "reverberance": 50,
            # }

        ],
        description="Sox effects parameters"
    )

    @field_serializer('sox_effects')
    def serialize(self, value: List[Dict[str, Union[int, float, str]]]) -> List[List[str]]:
        # return list of lists of strings for sox_effects, e.g. [['silence', '1', '0.2', '1%', '-1', '0.2', '1%']]
        return [[str(i) for i in effect.values()] for effect in value]


if __name__ == '__main__':
    print(_AudioConfig().model_dump())
    print(_AudioConfig().model_dump()['sox_effects'])
