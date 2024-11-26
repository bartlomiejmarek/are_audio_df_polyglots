import numpy as np
import torch
from scipy import signal
import copy

from configuration.rawboost_config import _RawboostConfig
from df_logger import main_logger

"""
___author__ = "Massimiliano Todisco, Hemlata Tak"
__email__ = "{todisco,tak}@eurecom.fr"
"""

'''
   Hemlata Tak, Madhu Kamble, Jose Patino, Massimiliano Todisco, Nicholas Evans.
   RawBoost: A Raw Data Boosting and Augmentation Method applied to Automatic Speaker Verification Anti-Spoofing.
   In Proc. ICASSP 2022, pp:6382--6386.
'''


def rand_range(x1, x2, integer):
    y = np.random.uniform(low=x1, high=x2, size=(1,))
    if integer:
        return int(y)
    return y


def norm_wav(x, always):
    if always:
        return x / np.amax(abs(x))
    elif np.amax(abs(x)) > 1:
        return x / np.amax(abs(x))
    return x


def gen_notch_coeffs(n_bands, min_f, max_f, min_bw, max_bw, min_coeff, max_coeff, min_g, max_g, fs):
    b = 1
    for i in range(0, n_bands):
        fc = rand_range(min_f, max_f, 0)
        bw = rand_range(min_bw, max_bw, 0)
        c = rand_range(min_coeff, max_coeff, 1)

        if c / 2 == int(c / 2):
            c = c + 1
        f1 = fc - bw / 2
        f2 = fc + bw / 2
        if f1 <= 0:
            f1 = 1 / 1000
        if f2 >= fs / 2:
            f2 = fs / 2 - 1 / 1000
        b = np.convolve(signal.firwin(c, [float(f1), float(f2)], window='hamming', fs=fs), b)

    G = rand_range(min_g, max_g, 0);
    _, h = signal.freqz(b, 1, fs=fs)
    return pow(10, G / 20) * b / np.amax(abs(h))


def filter_fir(x, b):
    N = b.shape[0] + 1
    xpad = np.pad(x, (0, N), 'constant')
    y = signal.lfilter(b, 1, xpad)
    return y[int(N / 2):int(y.shape[0] - N / 2)]


# Linear and non-linear convolution noise
def l_nl_convolutive_noise(x, n_f, n_bands, min_f, max_f, min_bw, max_bw, min_coeff, max_coeff, min_g, max_g,
                           min_bias_lin_non_lin, max_bias_lin_non_lin, fs):
    y = [0] * x.shape[0]
    for i in range(0, n_f):
        if i == 1:
            min_g = min_g - min_bias_lin_non_lin;
            max_g = max_g - max_bias_lin_non_lin;
        b = gen_notch_coeffs(n_bands, min_f, max_f, min_bw, max_bw, min_coeff, max_coeff, min_g, max_g, fs)
        y = y + filter_fir(np.power(x, (i + 1)), b)
    y = y - np.mean(y)
    return normalize_wav(y, 0)


# Impulsive signal dependent noise
def isd_additive_noise(x, p, g_sd):
    beta = rand_range(0, p, 0)

    y = copy.deepcopy(x)
    x_len = x.shape[0]
    n = int(x_len * (beta / 100))
    p = np.random.permutation(x_len)[:n]
    f_r = np.multiply(((2 * np.random.rand(p.shape[0])) - 1), ((2 * np.random.rand(p.shape[0])) - 1))
    r = g_sd * x[p] * f_r
    y[p] = (x[p] + r).float() if isinstance(x[p] + r, torch.Tensor) else (x[p] + r)
    return normalize_wav(y, False)


# Stationary signal independent noise

def ssi_additive_noise(x, snr_min, snr_max, n_bands, min_f, max_f, min_bw, max_bw, min_coeff, max_coeff, min_g, max_g,
                       fs):
    noise = np.random.normal(0, 1, x.shape[0])
    b = gen_notch_coeffs(n_bands, min_f, max_f, min_bw, max_bw, min_coeff, max_coeff, min_g, max_g, fs)
    noise = filter_fir(noise, b)
    noise = normalize_wav(noise, 1)
    SNR = rand_range(snr_min, snr_max, 0)
    noise = noise / np.linalg.norm(noise, 2) * np.linalg.norm(x, 2) / 10.0 ** (0.05 * SNR)
    return x + noise


def normalize_wav(x, always=False):
    max_val = abs(x).max()
    if always or max_val > 1:
        x = x / max_val
    return x


def process_rawboost_feature(feature, sr, rawboost_config: _RawboostConfig):
    def apply_impulsive_noise(x):
        return isd_additive_noise(
            x=x,
            p=rawboost_config.p,
            g_sd=rawboost_config.g_sd
        )

    def apply_colored_additive_noise(x):
        return ssi_additive_noise(
            x=x,
            snr_min=rawboost_config.snr_min,
            snr_max=rawboost_config.snr_max,
            n_bands=rawboost_config.n_bands,
            min_f=rawboost_config.min_f,
            max_f=rawboost_config.max_f,
            min_bw=rawboost_config.min_bw,
            max_bw=rawboost_config.max_bw,
            min_coeff=rawboost_config.min_coeff,
            max_coeff=rawboost_config.max_coeff,
            min_g=rawboost_config.min_g,
            max_g=rawboost_config.max_g,
            fs=sr
        )

    def apply_convolutive_noise(x):
        return l_nl_convolutive_noise(
            x=x,
            n_f=rawboost_config.n_f,
            n_bands=rawboost_config.n_bands,
            min_f=rawboost_config.min_f,
            max_f=rawboost_config.max_f,
            min_bw=rawboost_config.min_bw,
            max_bw=rawboost_config.max_bw,
            min_coeff=rawboost_config.min_coeff,
            max_coeff=rawboost_config.max_coeff,
            min_g=rawboost_config.min_g,
            max_g=rawboost_config.max_g,
            min_bias_lin_non_lin=rawboost_config.min_bias_lin_non_lin,
            max_bias_lin_non_lin=rawboost_config.max_bias_lin_non_lin,
            fs=sr
        )

    if rawboost_config.algo_id == 9:
        algo_id = np.random.randint(1, 8)

    else:
        algo_id = rawboost_config.algo_id

    if len(feature.shape) > 1:
        feature = feature.squeeze()

    # Data process by Convolutive noise (1st algo)
    if algo_id == 1:
        main_logger.debug("Applying Convolutive noise (RawBoost algo 1)")
        return apply_convolutive_noise(feature)

    # Data process by Impulsive noise (2nd algo)
    elif algo_id == 2:
        main_logger.debug("Applying Impulsive noise (RawBoost algo 2)")
        return apply_impulsive_noise(feature)

    # Data process by coloured additive noise (3rd algo)
    elif algo_id == 3:
        main_logger.debug("Applying Colored additive noise (RawBoost algo 3)")
        return apply_colored_additive_noise(feature)

    # Data process by all 3 algo. together in series (1+2+3)
    elif algo_id == 4:
        main_logger.debug("Applying Convolutive, Impulsive and Colored additive noise in series (RawBoost algo 4)")
        feature = apply_convolutive_noise(feature)
        feature = apply_impulsive_noise(feature)
        return apply_colored_additive_noise(feature)

        # Data process by 1st two algo. together in series (1+2)
    elif algo_id == 5:
        main_logger.debug("Applying Convolutive and Impulsive noise in series (RawBoost algo 5)")
        feature = apply_convolutive_noise(feature)
        return apply_impulsive_noise(feature)

        # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo_id == 6:
        main_logger.debug("Applying Convolutive and Colored additive noise in series (RawBoost algo 6)")
        feature = apply_impulsive_noise(feature)
        return apply_colored_additive_noise(feature)

        # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo_id == 7:
        main_logger.debug("Applying Impulsive and Colored additive noise in series (RawBoost algo 7)")
        feature = apply_impulsive_noise(feature)
        return apply_colored_additive_noise(feature)
        # Data process by 1st two algo. together in Parallel (1||2)
    elif algo_id == 8:
        main_logger.debug("Applying Convolutive and Impulsive noise in parallel (RawBoost algo 8)")
        feature1 = apply_convolutive_noise(feature)
        feature2 = apply_impulsive_noise(feature)
        return torch.from_numpy(normalize_wav(np.concatenate([feature1, feature2], 0)))

    # original data without Rawboost processing
    else:
        main_logger.debug("No RawBoost processing")
        return feature

