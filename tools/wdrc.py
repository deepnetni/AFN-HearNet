from matplotlib.pyplot import axis
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple
from scipy.signal import get_window
from librosa import stft, istft


def init_bands(nChannel: int) -> Tuple[np.ndarray, np.ndarray]:
    if nChannel == 4:
        channels = [0, 750, 1500, 3000, 8001]
        fc = [500, 1000, 2000, 4000]
    if nChannel == 6:
        channels = [0, 250, 625, 1375, 2500, 3500, 8001]
        fc = [250, 500, 1000, 2000, 3000, 4000]
    elif nChannel == 8:
        channels = [0, 250, 500, 750, 1250, 2500, 3500, 5000, 8001]
        fc = [125, 375, 625, 1000, 1870, 3000, 4250, 6500]
    elif nChannel == 12:
        channels = [0, 250, 375, 500, 750, 1125, 1500, 1875, 2625, 3375, 4250, 5625, 8001]
        fc = [250, 375, 500, 750, 1000, 1375, 1750, 2250, 3000, 3875, 4875, 6250]
    elif nChannel == 16:
        # fmt: off
        channels = [0, 250, 375, 500, 625, 750, 1000, 1250, 1625, 2000, 2375, 2875, 3500, 4250, 5125, 6125, 8001]
        fc = [250, 375, 500, 625, 750, 1000, 1125, 1375, 1750, 2125, 2625, 3125, 3875, 4625, 5500, 6625]
        # fmt: on
    elif nChannel == 124:
        # fmt: off
        channels = [0, 250, 375, 500, 625, 875, 1125, 1375, 1625, 1875, 2125, 2375, 2625, 2875, 3125,
                    3375, 3625, 3875, 4250, 4625, 5000, 5375, 5750, 6250, 8001]
        fc = [ 250, 375, 500, 625, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250,
               3500, 3750, 4000, 4375, 4750, 5125, 5500, 5875, 6625]
        # fmt: on
    else:
        raise RuntimeError(f"{nChannel} not supported.")

    return np.array(channels), np.array(fc)


def init_cur(nChannel: int) -> Tuple[np.ndarray, np.ndarray]:
    # TODO modify
    if nChannel == 4:
        k1 = [1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000]
        k2 = [0.60000, 0.60000, 0.60000, 0.60000, 0.60000, 0.60000, 0.60000, 0.60000]
        k3 = [0.33333, 0.33333, 0.33333, 0.33333, 0.33333, 0.33333, 0.33333, 0.33333]
        b1 = [10.0000, 15.0000, 20.0000, 20.0000, 20.0000, 20.0000, 20.0000, 20.0000]
        b2 = [28.0000, 33.0000, 38.0000, 38.0000, 38.0000, 38.0000, 38.0000, 38.0000]
        b3 = [48.0000, 53.0000, 58.0000, 58.0000, 58.0000, 58.0000, 58.0000, 58.0000]
    if nChannel == 6:
        k1 = [1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000]
        k2 = [0.60000, 0.60000, 0.60000, 0.60000, 0.60000, 0.60000, 0.60000, 0.60000]
        k3 = [0.33333, 0.33333, 0.33333, 0.33333, 0.33333, 0.33333, 0.33333, 0.33333]
        b1 = [10.0000, 15.0000, 20.0000, 20.0000, 20.0000, 20.0000, 20.0000, 20.0000]
        b2 = [28.0000, 33.0000, 38.0000, 38.0000, 38.0000, 38.0000, 38.0000, 38.0000]
        b3 = [48.0000, 53.0000, 58.0000, 58.0000, 58.0000, 58.0000, 58.0000, 58.0000]
    elif nChannel == 8:
        k1 = [1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000]
        k2 = [0.60000, 0.60000, 0.60000, 0.60000, 0.60000, 0.60000, 0.60000, 0.60000]
        k3 = [0.33333, 0.33333, 0.33333, 0.33333, 0.33333, 0.33333, 0.33333, 0.33333]
        b1 = [10.0000, 15.0000, 20.0000, 20.0000, 20.0000, 20.0000, 20.0000, 20.0000]
        b2 = [28.0000, 33.0000, 38.0000, 38.0000, 38.0000, 38.0000, 38.0000, 38.0000]
        b3 = [48.0000, 53.0000, 58.0000, 58.0000, 58.0000, 58.0000, 58.0000, 58.0000]
    elif nChannel == 12:
        k1 = [1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000]
        k2 = [0.60000, 0.60000, 0.60000, 0.60000, 0.60000, 0.60000, 0.60000, 0.60000]
        k3 = [0.33333, 0.33333, 0.33333, 0.33333, 0.33333, 0.33333, 0.33333, 0.33333]
        b1 = [10.0000, 15.0000, 20.0000, 20.0000, 20.0000, 20.0000, 20.0000, 20.0000]
        b2 = [28.0000, 33.0000, 38.0000, 38.0000, 38.0000, 38.0000, 38.0000, 38.0000]
        b3 = [48.0000, 53.0000, 58.0000, 58.0000, 58.0000, 58.0000, 58.0000, 58.0000]
    elif nChannel == 16:
        # fmt: off
        k1 = [1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000]
        k2 = [0.60000, 0.60000, 0.60000, 0.60000, 0.60000, 0.60000, 0.60000, 0.60000]
        k3 = [0.33333, 0.33333, 0.33333, 0.33333, 0.33333, 0.33333, 0.33333, 0.33333]
        b1 = [10.0000, 15.0000, 20.0000, 20.0000, 20.0000, 20.0000, 20.0000, 20.0000]
        b2 = [28.0000, 33.0000, 38.0000, 38.0000, 38.0000, 38.0000, 38.0000, 38.0000]
        b3 = [48.0000, 53.0000, 58.0000, 58.0000, 58.0000, 58.0000, 58.0000, 58.0000]
        # fmt: on
    elif nChannel == 124:
        # fmt: off
        k1 = [1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000]
        k2 = [0.60000, 0.60000, 0.60000, 0.60000, 0.60000, 0.60000, 0.60000, 0.60000]
        k3 = [0.33333, 0.33333, 0.33333, 0.33333, 0.33333, 0.33333, 0.33333, 0.33333]
        b1 = [10.0000, 15.0000, 20.0000, 20.0000, 20.0000, 20.0000, 20.0000, 20.0000]
        b2 = [28.0000, 33.0000, 38.0000, 38.0000, 38.0000, 38.0000, 38.0000, 38.0000]
        b3 = [48.0000, 53.0000, 58.0000, 58.0000, 58.0000, 58.0000, 58.0000, 58.0000]
        # fmt: on
    else:
        raise RuntimeError(f"{nChannel} not supported.")

    # 3, nChannel
    return np.stack([k1, k2, k3], axis=0), np.stack([b1, b2, b3], axis=0)


@dataclass
class WDRCconf:
    # Calibrated offset according to microphone gain and actual sound pressure level.
    nframe: int = 256
    nhop: int = nframe // 2
    nbin: int = nframe // 2 + 1
    fs: int = 16000
    # frequency resolution of fft
    freqs: np.ndarray = np.arange(0, fs / 2 + 1, fs / nframe)
    win = np.sqrt(get_window("hann", nframe, fftbins=True))

    TK1: float = 45.0
    TK2: float = 75.0
    attack_smooth: float = 0.5
    release_smooth: float = 0.98

    # 0: for sub-band, 1: for full-band when computing the spl_in dB.
    SPL_mode: int = 0
    nChannel: int = 8

    channels: np.ndarray = init_bands(nChannel)[0]
    fc: np.ndarray = init_bands(nChannel)[1]
    curve_kb = init_cur(nChannel)

    SPL_offset: float = 95.1079128 - 15  # dB unit
    SPK_offset: np.ndarray = np.zeros(nChannel)  # calibration factors for the speaker and ADC
    MPO: float = 100  # maximum SPL output

    def __post_init__(self):
        # the following operation will modify the original data, conf.curve_kb
        k, b = self.curve_kb
        # TODO check
        b[..., -1] = b[..., -1] - 5 if self.SPL_mode == 0 else b[..., -1]


wdrc_conf = WDRCconf(SPL_mode=1)
# print(wdrc_conf.curve_kb[-1], wdrc_conf.SPL_mode)


def compute_SPL(conf: WDRCconf, xk: np.ndarray):
    """
    xk: b,2,t,f if real or b,t,f if complex
    """
    if np.iscomplexobj(xk):
        # b,t,f->b,2,t,f
        xk = np.stack([xk.real, xk.imag], axis=1)

    pow_l = []
    if conf.SPL_mode == 0:  # sub-band
        for i, (low, high) in enumerate(zip(conf.channels[:-1], conf.channels[1:])):
            # idx=(x,) if only one component, making sure the xk[..., idx].shape to b,2,t,L
            idx = np.where((conf.freqs >= low) & (conf.freqs < high))[0]
            pow = np.sum(xk[..., idx] ** 2, axis=(1, 3))  # B,T
            pow_l.append(pow / len(idx))  # B,T,C
    else:  # full-band
        pow = np.sum(xk**2, axis=(1, 3))  # B,T
        pow_l.append(pow / xk.shape[-1])  # B,T,1

    pow_c = np.stack(pow_l, axis=-1)
    return 10 * np.log10(pow_c + 1e-10) + conf.SPL_offset  # B,T,C


def apply_subbands_gain(conf, xk, gain, decay=None):
    """
    xk: spectrum with shape b,2,t,f or complex type with shape b,t,f.
    gain: gain of each subbands, with shape b,t,c;
    decay: the decay vector, b,f(65)
    """
    if np.iscomplexobj(xk):
        # b,t,f->b,2,t,f
        xk = np.stack([xk.real, xk.imag], axis=1)

    gain = gain[:, None, ...]  # b,t,c -> b,1,t,c

    # b,2,t,f
    G = np.ones_like(xk, dtype=np.float32)

    for i, (low, high) in enumerate(zip(conf.channels[:-1], conf.channels[1:])):
        idx = np.where((conf.freqs >= low) & (conf.freqs < high))[0]
        # gain b,1,t,1 copy to b,2,t,n
        G[..., idx] = gain[..., (i,)]  #

    if decay is not None:
        # b,2,t,f x b,1,t,f x b,1,1,f
        xk_out = xk * G * decay[:, None, None, :]
    else:
        xk_out = xk * G

    return xk_out


def spl2wave(conf, spl) -> np.ndarray:
    """
    spl: b,t,c, range about [30-100] dB
    """
    # b,t
    spl = spl.mean(-1) * 0.01
    # spl = spl.repeat_interleave([])
    spl = spl.repeat(conf.nhop, 1)

    return spl


def wdrc_process(xk: np.ndarray, conf: WDRCconf = wdrc_conf):
    """
    xk: b,2,t,f if real or b,t,f if complex
        or b,t if waveform.

    return: wdrc_output, smoothed spl_in
    """
    if xk.ndim < 3:  # waveform
        xk = xk[None, :] if xk.ndim == 1 else xk

        xk = stft(
            xk,  # B,T,
            win_length=conf.nframe,
            n_fft=conf.nframe,
            hop_length=conf.nhop,
            window=conf.win,
            center=False,
        )  # output shape B,F,T
        # xkk = np.stack([xk.real, xk.imag], axis=1)
        xk = xk.transpose(0, 2, 1)  # b,f,t -> b,t,f
        is_waveform = True
    else:
        is_waveform = False

    if np.iscomplexobj(xk):
        # b,t,f->b,2,t,f
        xk = np.stack([xk.real, xk.imag], axis=1)

    # NOTE xk: B,2,T,F
    spl_in = compute_SPL(conf, xk)  # B,T,C
    # smooth the SPL
    spl_buff = np.zeros((spl_in.shape[0], spl_in.shape[-1]))  # B,C
    spl_l = []
    for t in range(spl_in.shape[1]):
        SPL_t = spl_in[:, t, ...]  # B,C
        # attack
        spl_tmp = np.where(
            SPL_t > spl_buff,
            SPL_t * (1 - conf.attack_smooth) + spl_buff * conf.attack_smooth,
            SPL_t * (1 - conf.release_smooth) + spl_buff * conf.release_smooth,
        )
        spl_tmp[spl_tmp < 0] = 0
        spl_l.append(spl_tmp)
        spl_buff = SPL_t
    spl_in = np.stack(spl_l, axis=1)  # B,T,C

    # NOTE The spl_in for each sub-band is the same calculating from the full-band region;
    # but due to differing gain curves, the resulting SPLout may not be the same.

    kn, bn = conf.curve_kb
    # apply the gain curve coeff
    spl_out = np.where(spl_in < conf.TK1, kn[0, :] * spl_in + bn[0, :], spl_in)
    spl_out = np.where(
        (spl_in > conf.TK1) & (spl_in < conf.TK2), kn[1, :] * spl_in + bn[1, :], spl_out
    )
    spl_out = np.where(spl_in >= conf.TK2, kn[2, :] * spl_in + bn[2, :], spl_out)
    spl_out[spl_out > conf.MPO] = conf.MPO

    gain_dB = spl_out - spl_in + conf.SPK_offset  # B,T,C
    gain = 10 ** (gain_dB / 20)  # the gain applied to the spectrum

    # b,2,t,f
    x_out = apply_subbands_gain(conf, xk, gain)

    if is_waveform:  # return to time domain
        # b,t,f
        x_out = x_out[:, 0, ...] + x_out[:, 1, ...] * 1j
        x_out[..., 0] = 0
        x_out = x_out.transpose(0, 2, 1)  # b,t,f -> b,f,t
        x_out = istft(
            x_out,
            win_length=conf.nframe,
            n_fft=conf.nframe,
            hop_length=conf.nhop,
            window=conf.win,
            center=False,
        )  # output shape B,T

    return x_out.squeeze(), spl2wave(conf, spl_in).squeeze()


def mix_white_noise(x: np.ndarray, snr=15):
    assert x.ndim == 1

    def rms(audio, db=False):
        audio = np.asarray(audio)
        rms_value = np.sqrt(np.mean(audio**2))
        if db:
            return 20 * np.log10(rms_value + np.finfo(float).eps)
        else:
            return rms_value

    noise = np.random.randn(len(x))
    scale = rms(x) / (rms(noise) * 10 ** (snr / 20))
    noise = scale * noise

    mix = x + noise
    scaler = mix.max()
    if scaler >= 1:
        mix = mix / scaler
        x = x / scaler

    return mix, x


if __name__ == "__main__":
    import soundfile as sf

    # inp = np.random.randn(1, 16000)
    # inp, fs = sf.read("/home/deepni/datasets/dnsc/dnsc_clean/clean_fileid_0.wav")
    inp, fs = sf.read("./verify_audio/wdrc_calibration_audio.wav")
    mix, inp = mix_white_noise(inp)

    out, spl = wdrc_process(inp, wdrc_conf)
    N = spl.shape[-1]
    # print(out.shape, spl.shape, N, mix.shape)

    # out = np.stack([mix[..., :N], spl], axis=-1)
    out = np.stack([inp[:N], out[:N], spl], axis=-1)
    sf.write("out.wav", out, fs)
    # sf.write("inp.wav", mix, fs)
