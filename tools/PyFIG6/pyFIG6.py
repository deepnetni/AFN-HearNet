import sys

# sys.path.append(__file__.rsplit("/", 4)[0])

import numpy as np
from librosa import istft, stft
from scipy.interpolate import interp1d
from scipy.signal import get_window
from utils.vad import VAD


def HL_LinearFitting(src, freq, ht, hearingNum):
    """Hearing level
    ht: hearing threshold;
    """

    for i in range(hearingNum - 1):
        if src >= freq[i] and src <= freq[i + 1]:
            if freq[i] == freq[i + 1]:
                return ht[i]
            else:
                k = (ht[i + 1] - ht[i]) / (freq[i + 1] - freq[i])
                b = ht[i] - k * freq[i]
                return k * src + b


def fig6_curve(audiogram_f: np.ndarray, audiogram_ht, ChannelNum_fc):
    """the audiogram_ht is the hearing threshold level, unit dB, at the audiogram feature points.
    audiogram_f: audiogram feature points
    audiogram_ht: hearing threshold.

    Return: k, b with shape [ChannelNum, 3]

    """

    audiogram_ft = np.array([0, *audiogram_f])
    audiogram_ht = np.array([0, *audiogram_ht])
    ChannelNum = len(ChannelNum_fc)
    htn = np.zeros(ChannelNum)
    for i in range(ChannelNum):
        htn[i] = HL_LinearFitting(
            ChannelNum_fc[i], audiogram_ft, audiogram_ht, len(audiogram_ht)
        )

    # NOTE, b is the gain
    k = np.zeros((ChannelNum, 3))  # 3 for input dB level: 40, 65, 90
    b = np.zeros((ChannelNum, 3))
    tklin = 40  # target gain level input
    tkhin = 60

    # NOTE compute cure coeffs of each channel
    for i in range(ChannelNum):
        ht = htn[i]
        k[i, 0] = 1
        if ht < 20:  # 0-20
            b[i, 0] = 0
            k[i, 1] = 1
            b[i, 1] = 0
            k[i, 2] = 1
            b[i, 2] = 0
        elif ht < 40:  # 20-40
            # ----- 40dB low level
            ig40 = ht - 20
            splout40 = 40 + ig40
            b[i, 0] = splout40 - 40
            tklout = tklin + b[i, 0]

            # ----- 65dB comfortable level
            ig65 = 0.6 * (ht - 20)
            splout65 = 65 + ig65
            # k[i,1] = (splout65 - tklout) / (65 - tklin)
            # b[i,1] = (tklout * 65 - splout65 * tklin) / (65 - tklin)
            k[i, 1] = (splout65 - tklout) / (65 - tklin) - 1.5
            b[i, 1] = (tklout * 65 - splout65 * tklin) / (65 - tklin) - 30
            tkhout = k[i, 1] * tkhin + b[i, 1]

            # ----- 90dB high level
            ig95 = 0
            splout95 = 95 + ig95
            k[i, 2] = (splout95 - tkhout) / (95 - tkhin)
            b[i, 2] = (tkhout * 95 - splout95 * tkhin) / (95 - tkhin)
        elif ht < 60:  # 40-60
            ig40 = ht - 20
            splout40 = 40 + ig40
            b[i, 0] = splout40 - 40
            tklout = tklin + b[i, 0]

            ig65 = 0.6 * (ht - 20)
            splout65 = 65 + ig65
            # k[i,2] = (splout65 - tklout) / (65 - tklin)
            # b[i,2] = (tklout * 65 - splout65 * tklin) / (65 - tklin)
            k[i, 1] = (splout65 - tklout) / (65 - tklin) - 1.5
            b[i, 1] = (tklout * 65 - splout65 * tklin) / (65 - tklin) - 30
            tkhout = k[i, 1] * tkhin + b[i, 1]

            ig95 = 0.1 * (ht - 40) ** 1.4

            splout95 = 95 + ig95
            k[i, 2] = (splout95 - tkhout) / (95 - tkhin)
            b[i, 2] = (tkhout * 95 - splout95 * tkhin) / (95 - tkhin)
        else:  # > 60 dBHL
            ig40 = ht - 20 - 0.5 * (ht - 60)

            splout40 = 40 + ig40
            b[i, 0] = splout40 - 40
            tklout = tklin + b[i, 0]

            ig65 = 0.8 * ht - 23
            splout65 = 65 + ig65
            # k[i,2] = (splout65 - tklout) / (65 - tklin)
            # b[i,2] = (tklout * 65 - splout65 * tklin) / (65 - tklin)
            k[i, 1] = (splout65 - tklout) / (65 - tklin) - 1.5
            b[i, 1] = (tklout * 65 - splout65 * tklin) / (65 - tklin) - 30
            tkhout = k[i, 1] * tkhin + b[i, 1]

            ig95 = 0.1 * (ht - 40) ** 1.4

            splout95 = 95 + ig95
            k[i, 2] = (splout95 - tkhout) / (95 - tkhin)
            b[i, 2] = (tkhout * 95 - splout95 * tkhin) / (95 - tkhin)

    return k, b


def compute_subbands_SPL(xk, freqs, subbands, SPL_offset):
    """
    xk, the spectrum, complex with shape b,t,f or b,2,t,f
    freqs, the resolution of fft
    subbands,
    """
    if np.iscomplexobj(xk):
        # b,t,f->b,2,t,f
        xk = np.stack([xk.real, xk.imag], axis=1)

    pow_l = []
    for i, (low, high) in enumerate(zip(subbands[:-1], subbands[1:])):
        idx = np.where((freqs >= low) & (freqs < high))[0]
        pow = np.sum(xk[..., idx] ** 2, axis=(1, 3))  # B,T
        # TODO check whether divide the sub-band frequency points.
        # pow_l.append(pow / len(idx))
        pow_l.append(pow)

    pow_c = np.stack(pow_l, axis=-1)
    return 20 * np.log10(np.sqrt(pow_c) + 1e-7) + SPL_offset  # B,T,C


def apply_subbands_gain(xk, gain, freqs, subbands, decay=None):
    """
    xk: spectrum with shape b,2,t,f or complex type with shape b,t,f.
    gain: gain of each subbands, with shape b,t,c;
    freqs: frequency component of spectrum, f.
    subbands: channel+1, [start, end) region of each subbands, e.g., [125,250,500,...]
    decay: the decay vector, b,f(65)
    """
    if np.iscomplexobj(xk):
        # b,t,f->b,2,t,f
        xk = np.stack([xk.real, xk.imag], axis=1)

    gain = gain[:, None, ...]  # b,t,c -> b,1,t,c

    # b,2,t,f
    G = np.ones_like(xk, dtype=np.float32)

    for i, (low, high) in enumerate(zip(subbands[:-1], subbands[1:])):
        idx = np.where((freqs >= low) & (freqs < high))[0]
        # gain b,1,t,1 copy to b,2,t,n
        # G[..., idx] = np.repeat(gain[..., i][..., None], len(idx), axis=-1)
        G[..., idx] = gain[..., i][..., None]  #

    if decay is not None:
        # b,2,t,f x b,1,t,f x b,1,1,f
        xk_out = xk * G * decay[:, None, None, :]
    else:
        xk_out = xk * G

    return xk_out


def FIG6_compensation(HL, inp, fs=16000, nframe=128, nhop=64, SPL_off=None):
    """
    HL: hearing level at [250, 500, 1000, 2000, 4000, 8000]Hz, dB HL.
    inp: signal to compensation. shape, B,T
    fs: sample rate.
    nframe: length of FFT.
    nhop: block length.

    return B,T
    """

    if isinstance(HL, list):
        HL = np.array(HL)

    inp = inp[None, :] if inp.ndim == 1 else inp

    win = get_window("hann", nframe, fftbins=True)
    win = np.sqrt(win)
    nbin = nhop + 1
    ua = 0.1  # params to adjust the attack time
    ur = 0.95  # params to adjust the release time
    fstep = fs // nframe
    freso = np.arange(0, fs // 2 + 1, fstep)
    assert fstep % 125 == 0

    HL_avg = HL[:4].mean()
    HL_Flag = 0 if HL_avg < 60 else 1
    TK = [40, 60]
    MPO = 110

    ChannelNum = 16
    if ChannelNum == 4:
        SPL_offset = 95.1002986 * np.ones(ChannelNum)
        ChannelNum_ft = np.array([0, 750, 1500, 3000, 8001])
        ChannelNum_fc = np.array([500, 1000, 2000, 4000])
    elif ChannelNum == 6:
        SPL_offset = (
            94.9133059 * np.ones(ChannelNum)
            if SPL_off is None
            else SPL_off * np.ones(ChannelNum)
        )
        ChannelNum_ft = np.array([0, 250, 625, 1375, 2500, 3500, 8001])
        ChannelNum_fc = np.array([250, 500, 1000, 2000, 3000, 4000])
    elif ChannelNum == 8:
        SPL_offset = 95.1079128 * np.ones(ChannelNum)
        ChannelNum_ft = np.array([0, 250, 500, 750, 1375, 2500, 3500, 4875, 8001])
        ChannelNum_fc = np.array([250, 500, 750, 1125, 1750, 2500, 4000, 6000])
    elif ChannelNum == 12:
        SPL_offset = 95.31420495 * np.ones(ChannelNum)
        ChannelNum_ft = np.array(
            [0, 250, 375, 500, 750, 1125, 1500, 1875, 2625, 3375, 4250, 5625, 8001]
        )
        ChannelNum_fc = np.array(
            [250, 375, 500, 750, 1000, 1375, 1750, 2250, 3000, 3875, 4875, 6250]
        )
    elif ChannelNum == 16:
        SPL_offset = (
            96.7119344 * np.ones(ChannelNum)
            if SPL_off is None
            else SPL_off * np.ones(ChannelNum)
        )

        # fmt: off
        ChannelNum_ft = np.array([0, 250, 375, 500, 625, 750, 1000, 1250, 1625, 2000, 2375, 2875, 3500, 4250, 5125, 6125, 8001])
        ChannelNum_fc = np.array([250, 375, 500, 625, 750, 1000, 1125, 1375, 1750, 2125, 2625, 3125, 3875, 4625, 5500, 6625])
        # fmt: on
    else:
        raise RuntimeError("channel num is not supported.")

    # NOTE fitting based on the audiogram
    audiogram_f = np.array(
        [125, 250, 500, 1000, 2000, 4000, 8001]
    )  # audiogram feature points
    audiogram_ht = np.array(
        [HL[0], *HL]
    )  # hearing threshold corresponding to each feature points dB

    # if HL_Flag == 1:
    #     # determine the attenuation vector according to the degree of HL value.
    #     if HL_avg > 20 and HL_avg <= 40:
    #         Hearing_Loss = np.array([0, 0, 0, 0, -10, -20, 0])
    #     elif HL_avg > 40 and HL_avg <= 60:
    #         Hearing_Loss = np.array([0, 0, -10, -10, -20, -30, -10])
    #     elif HL_avg > 60:
    #         Hearing_Loss = np.array([0, 0, -10, -20, -30, -40, -30])
    #     else:
    #         Hearing_Loss = np.zeros(7)

    #     Hearing_Loss = 10.0 ** (Hearing_Loss / 10)
    #     idx = audiogram_f / fstep  # ignore the 0-th point
    #     interp = interp1d(idx, Hearing_Loss)
    #     HL_ext = interp(np.arange(1, nbin))
    #     HL_ext = np.nan_to_num(HL_ext, nan=0.0)

    #     # TODO check, process the 0-th component
    #     HL_ext = np.array([1, *HL_ext])  # 65
    # else:  # HL_Flag == 0
    #     HL_ext = np.ones(nbin)  # 65,
    # HL_ext = HL_ext[None, ...]

    # C,3
    kn, bn = fig6_curve(audiogram_f, audiogram_ht, ChannelNum_fc)

    xk = stft(
        inp,  # B,T,
        win_length=nframe,
        n_fft=nframe,
        hop_length=nhop,
        window=win,
        center=False,
    )  # output shape B,F,T
    # xkk = np.stack([xk.real, xk.imag], axis=1)
    xk = xk.transpose(0, 2, 1)  # b,f,t -> b,t,f

    # DEBUG
    # xk = np.ones((1, 24, 65)) / 10 + np.ones((1, 24, 65)) / 10 * 1j

    # if HL_Flag == 1:
    #     HearingLoss_xk = xk * HL_ext[None, :]

    spl_in = compute_subbands_SPL(xk, freso, ChannelNum_ft, SPL_offset)
    # spl_in_buff = np.concatenate([np.zeros((*spl_in.shape[:2], 1)), spl_in[..., :-1]], axis=-1)
    splBuff = np.zeros((spl_in.shape[0], ChannelNum))
    spl_l = []

    # NOTE WDRC, wide dynamic range compression
    # for t in range(spl_in.shape[1]):
    #     splc_l = []
    #     for c in range(ChannelNum):
    #         splIn = spl_in[:, t, c]  # B,
    #         splIn = np.where(
    #             splIn > splBuff[:, c],
    #             splIn * (1 - ua) + splBuff[:, c] * ua,
    #             splIn * (1 - ur) + splBuff[:, c] * ur,
    #         )
    #         splIn[splIn < 0] = 0
    #         splBuff[:, c] = splIn
    #         splc_l.append(splIn)
    #     spl_l.append(np.stack(splc_l, axis=-1))  # B,C
    # spl_in = np.stack(spl_l, axis=1)  # B,T,C

    for t in range(spl_in.shape[1]):
        splIn = spl_in[:, t, :]  # B,C
        splIn = np.where(
            splIn > splBuff,
            splIn * (1 - ua) + splBuff * ua,
            splIn * (1 - ur) + splBuff * ur,
        )
        splIn[splIn < 0] = 0
        splBuff = splIn
        spl_l.append(splIn)  # B,C

    spl_in = np.stack(spl_l, axis=1)  # B,T,C

    spl_out = np.where(spl_in < TK[0], kn[:, 0] * spl_in + bn[:, 0], spl_in)
    spl_out = np.where(
        (spl_in > TK[0]) & (spl_in < TK[1]), kn[:, 1] * spl_in + bn[:, 1], spl_out
    )
    spl_out = np.where(spl_in >= TK[1], kn[:, 2] * spl_in + bn[:, 2], spl_out)
    spl_out[spl_out > MPO] = MPO

    gain_dB = spl_out - spl_in  # B,T,C
    gain = 10 ** (gain_dB / 20)  # the gain applied to the spectrum

    meta = dict(spl_in=spl_in, gain=gain_dB)

    if ChannelNum == nbin:
        # b,t,f * b,t,c
        xk_g = xk * gain
    else:
        # b,2,t,f
        # xk_g = apply_subbands_gain(xk, gain, freso, ChannelNum_ft, HL_ext)
        xk_g = apply_subbands_gain(xk, gain, freso, ChannelNum_ft)

    # b,t,f
    xk_g = xk_g[:, 0, ...] + xk_g[:, 1, ...] * 1j
    xk_g[..., 0] = 0
    xk = xk_g.transpose(0, 2, 1)  # b,t,f -> b,f,t
    x = istft(
        xk,
        win_length=nframe,
        n_fft=nframe,
        hop_length=nhop,
        window=win,
        center=False,
    )  # output shape B,T
    return x.squeeze(), meta


def FIG6_compensation_vad(
    HL, inp, fs=16000, nframe=128, nhop=64, vad=None, ret_vad=False, SPL_off=None
):
    if vad is None:
        vad_detect = VAD(10, fs, level=2)
        vad_detect.reset()
        x_vad = np.ones_like(inp) * 0.95
        vad_lbl = vad_detect.vad_waves(inp)  # T,
        x_vad[: len(vad_lbl)] = vad_lbl
    else:
        x_vad = vad
    comp, meta = FIG6_compensation(HL, inp, fs, nframe, nhop, SPL_off)
    inp = inp[: len(comp)]
    x_vad = x_vad[: len(comp)]
    comp = np.where(x_vad > 0.5, comp, inp)

    if ret_vad:
        return comp, x_vad, meta
    else:
        return comp


if __name__ == "__main__":
    import soundfile as sf

    HL = [50, 60, 70, 75, 85, 95]
    # src = np.random.randn(1600)
    # src = np.ones(1600)
    src, fs = sf.read("/home/deepni/datasets/dns_wdrc/test/0_enlarge_nearend.wav")
    out = FIG6_compensation(HL, src)
    print(out.shape, src.shape)
    # sf.write("out.wav", out, fs)
    # sf.write("src.wav", src, fs)
