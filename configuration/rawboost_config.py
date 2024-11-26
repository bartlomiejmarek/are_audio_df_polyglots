from pydantic import BaseModel, Field, conint


class _RawboostConfig(BaseModel):
    algo_id: conint(ge=0, le=9) = Field(
        description="""Rawboost algorithms descriptions. 
                      0: No augmentation, 
                      1: LnL_convolutive_noise, 
                      2: ISD_additive_noise,
                      3: SSI_additive_noise, 
                      4: series algo (1+2+3), 
                      5: series algo (1+2), 
                      6: series algo (1+3), 
                      7: series algo (2+3), 
                      8: parallel algo (1+2).
                      9: random algo selection.
                      """
    )
    # # LnL_convolutive_noise parameters
    n_bands: int = Field(5,
                         description="Number of notch filters. The higher the number of bands, the more aggressive the distortion is. Default=5")
    min_f: int = Field(default=20, description="Minimum centre frequency [Hz] of notch filter. Default=20")
    max_f: int = Field(default=8000, description="Maximum centre frequency [Hz] (<sr/2) of notch filter. Default=8000")
    min_bw: int = Field(default=100, description="Minimum width [Hz] of filter. Default=100")
    max_bw: int = Field(default=1000, description="Maximum width [Hz] of filter. Default=1000")
    min_coeff: int = Field(default=10,
                           description="Minimum filter coefficients. More the filter coefficients, more ideal the filter slope. Default=10")
    max_coeff: int = Field(default=100,
                           description="Maximum filter coefficients. More the filter coefficients, more ideal the filter slope. Default=100")
    min_g: int = Field(default=0, description="Minimum gain factor of linear component. Default=0")
    max_g: int = Field(default=0, description="Maximum gain factor of linear component. Default=0")
    min_bias_lin_non_lin: int = Field(default=5,
                                      description="Minimum gain difference between linear and non-linear components. Default=5")
    max_bias_lin_non_lin: int = Field(default=20,
                                      description="Maximum gain difference between linear and non-linear components. Default=20")
    n_f: int = Field(default=5,
                     description="Order of the (non-)linearity where N_f=1 refers only to linear components. Default=5")

    # ISD_additive_noise parameters
    p: int = Field(default=10, description="Maximum number of uniformly distributed samples in [%]. Default=10")
    g_sd: int = Field(default=2, description="Gain parameters > 0. Default=2")

    # SSI_additive_noise parameters
    snr_min: int = Field(default=10, description="Minimum SNR value for coloured additive noise. Default=10")
    snr_max: int = Field(default=40, description="Maximum SNR value for coloured additive noise. Default=40")
