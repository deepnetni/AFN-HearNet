import numpy as np
import math
import matlab.engine
import librosa
from scipy import signal


def eb_CenterFreq(nchan, shift=-1.0):
    ### ERB刻度计算中心频率
    lowFreq = 80.0
    highFreq = 8000.0

    EarQ = 9.26449
    minBW = 24.7

    if shift != -1:
        k = 1
        A = 165.4
        a = 2.1
        xLow = (1 / a) * np.log10(k + (lowFreq / A))
        xHigh = (1 / a) * np.log10(k + (highFreq / A))
        xLow = xLow * (1 + shift)
        xHigh = xHigh * (1 + shift)
        lowFreq = A * (10 ** (a * xLow) - k)
        highFreq = A * (10 ** (a * xHigh) - k)

    cf = -(EarQ * minBW) + np.exp(
        np.array(range(1, int(nchan)), dtype="float32")
        * (-np.log(highFreq + EarQ * minBW) + np.log(lowFreq + EarQ * minBW))
        / (nchan - 1)
    ) * (highFreq + EarQ * minBW)
    cf = cf[::-1]
    cf = np.append(cf, highFreq)

    return np.array(cf, dtype="float32")


def eb_LossParameters(HL, cfreq):
    aud = np.array([250, 500, 1000, 2000, 4000, 6000], dtype="float32")
    nfilt = len(cfreq)

    fv = np.insert(aud, 0, cfreq[0])
    fv = np.append(fv, cfreq[nfilt - 1])

    HL_new = np.array(HL, dtype="float32")
    HL_new = np.insert(HL_new, 0, HL[0])
    HL_new = np.append(HL_new, HL[-1])

    loss = np.interp(cfreq, fv, HL_new)

    loss = np.maximum(loss, 0)

    CR = np.zeros(nfilt, dtype="float32")

    for index in range(nfilt):
        CR[index] = 1.25 + 2.25 * (index) / (nfilt - 1)

    maxOHC = 70 * (1 - (1.0 / CR))
    thrOHC = 1.25 * maxOHC

    attnOHC = np.zeros(nfilt, dtype="float32")
    attnIHC = np.zeros(nfilt, dtype="float32")

    for n in range(nfilt):
        if loss[n] < thrOHC[n]:
            attnOHC[n] = 0.8 * loss[n]
            attnIHC[n] = 0.2 * loss[n]
        else:
            attnOHC[n] = 0.8 * thrOHC[n]
            attnIHC[n] = 0.2 * thrOHC[n] + (loss[n] - thrOHC[n])

    BW = np.ones(nfilt, dtype="float32")

    BW = BW + (attnOHC / 50.0) + 2.0 * (attnOHC / 50.0) ** 6

    lowknee = attnOHC + 30

    upamp = 30 + 70.0 / CR
    CR = (100 - lowknee) / (upamp + attnOHC - lowknee)

    return attnOHC, BW, lowknee, CR, attnIHC


def Fig6_Amplification(HL, x, xsamp):
    eng = matlab.engine.start_matlab()
    eng.cd("./HASQI_wFIG6_Matlab/HASQI", nargout=0)

    ht_matlab = matlab.double(HL)

    matlab_fs1 = matlab.double([xsamp])

    matlab_audio = matlab.double(x.tolist())

    x_fig6 = eng.Fig6_Amplification(ht_matlab, matlab_audio, matlab_fs1, nargout=1)

    return np.array(x_fig6).squeeze()


def eb_Resamp24kHz(x, xsamp):
    fsamp = 24000
    y = librosa.resample(x, orig_sr=xsamp, target_sr=fsamp, res_type="polyphase")
    xRMS = np.sqrt(np.mean(x**2))  ## 输入信号的均方根值
    yRMS = np.sqrt(np.mean(y**2))  ## 重采样后信号的均方根值
    y = (xRMS / yRMS) * y

    return np.array(y), fsamp


def eb_InputAlign(x, y):
    nx = len(x)
    ny = len(y)
    nsamp = min(nx, ny)

    x_dmean = x[:nsamp] - x[:nsamp].mean()
    y_dmean = y[:nsamp] - y[:nsamp].mean()

    xy = np.correlate(x_dmean, y_dmean, "full")

    index = np.argmax(np.abs(xy))
    delay = nsamp - 1 - index
    fsamp = 24000
    delay = delay - 2 * fsamp / 1000

    if delay > 0:
        pad = np.zeros(delay, dtype="float32")
        y = np.concatenate((y[delay:ny], pad), axis=0)
    else:
        delay = int(-delay)
        pad = np.zeros(delay, dtype="float32")
        y = np.concatenate((pad, y[: ny - delay]), axis=0)

    xabs = np.abs(x)
    xmax = np.max(xabs)
    xthr = 0.001 * xmax

    for n in range(nx):
        if xabs[n] > xthr:
            nx0 = n
            break
    for n in range(nx)[::-1]:
        if xabs[n] > xthr:
            nx1 = n
            break
    if nx1 > ny:
        nx1 = ny
    xp = x[nx0 : (nx1 + 1)]
    yp = y[nx0 : (nx1 + 1)]

    return xp, yp


def eb_MiddleEar(x, fsamp):
    bLP, aLP = signal.butter(1, 5000 / (0.5 * fsamp), "lowpass")
    filtedData = signal.lfilter(bLP, aLP, x)
    bHP, aHP = signal.butter(2, 350 / (0.5 * fsamp), "highpass")
    xout = signal.lfilter(bHP, aHP, filtedData)

    return xout


def eb_GammatoneEnv2_origin(x, BWx, y, BWy, fs, cf):
    earQ = 9.26449
    minBW = 24.7
    ERB = minBW + (cf / earQ)
    nx = len(x)
    ny = len(y)
    nsamp = min(nx, ny)
    x = x[:nsamp]
    y = y[:nsamp]

    tpt = 2 * math.pi / fs
    tptBW = BWx * tpt * ERB * 1.019
    a = np.exp(-tptBW)
    a1 = 4.0 * a
    a2 = -6.0 * a * a
    a3 = 4.0 * a * a * a
    a4 = -a * a * a * a
    a5 = 4.0 * a * a
    gain = 2.0 * (1 - a1 - a2 - a3 - a4) / (1 + a1 + a5)

    npts = len(x)
    cn = np.cos(tpt * cf)
    sn = np.sin(tpt * cf)
    coscf = np.zeros(npts, dtype="float32")
    sincf = np.zeros(npts, dtype="float32")
    cold = 1
    sold = 0
    coscf[0] = cold
    sincf[0] = sold
    for n in range(1, npts):
        arg = cold * cn + sold * sn
        sold = sold * cn - cold * sn
        cold = arg
        coscf[n] = cold
        sincf[n] = sold

    coscf = np.ones(npts, dtype="float32")
    sincf = np.ones(npts, dtype="float32")

    # ureal = signal.filtfilt([1, a1, a5], [1, -a1, -a2, -a3, -a4], x * coscf)
    ureal = signal.lfilter([1, a1, a5], [1, -a1, -a2, -a3, -a4], x * coscf)
    uimag = signal.lfilter([1, a1, a5], [1, -a1, -a2, -a3, -a4], x * sincf)
    envx = gain * np.sqrt(ureal * ureal + uimag * uimag + 1e-8)

    tptBW = BWy * tpt * ERB * 1.019
    a = np.exp(-tptBW)
    a1 = 4.0 * a
    a2 = -6.0 * a * a
    a3 = 4.0 * a * a * a
    a4 = -a * a * a * a
    a5 = 4.0 * a * a
    gain = 2.0 * (1 - a1 - a2 - a3 - a4) / (1 + a1 + a5)

    ureal = signal.lfilter([1, a1, a5], [1, -a1, -a2, -a3, -a4], y * coscf)
    uimag = signal.lfilter([1, a1, a5], [1, -a1, -a2, -a3, -a4], y * sincf)

    envy = gain * np.sqrt(ureal * ureal + uimag * uimag + 1e-8)

    return envx, envy


def eb_GammatoneEnv2(x, BWx, y, BWy, fs, cf, coscf, sincf):
    earQ = 9.26449
    minBW = 24.7
    ERB = minBW + (cf / earQ)
    nx = len(x)
    ny = len(y)
    nsamp = min(nx, ny)
    x = x[:nsamp]
    y = y[:nsamp]

    tpt = 2 * math.pi / fs
    tptBW = BWx * tpt * ERB * 1.019
    a = np.exp(-tptBW)
    a1 = 4.0 * a
    a2 = -6.0 * a * a
    a3 = 4.0 * a * a * a
    a4 = -a * a * a * a
    a5 = 4.0 * a * a
    gain = 2.0 * (1 - a1 - a2 - a3 - a4) / (1 + a1 + a5)

    # npts = len(x)
    # cn = np.cos(tpt*cf)
    # sn = np.sin(tpt*cf)
    # coscf = np.zeros(npts, dtype="float32")
    # sincf = np.zeros(npts, dtype="float32")
    # cold = 1
    # sold = 0
    # coscf[0] = cold
    # sincf[0] = sold
    # for n in range(1, npts):
    #     arg = cold * cn + sold * sn
    #     sold = sold * cn - cold * sn
    #     cold = arg
    #     coscf[n] = cold
    #     sincf[n] = sold
    #
    # coscf = np.ones(npts, dtype="float32")
    # sincf = np.ones(npts, dtype="float32")

    # ureal = signal.filtfilt([1, a1, a5], [1, -a1, -a2, -a3, -a4], x * coscf)
    ureal = signal.lfilter([1, a1, a5], [1, -a1, -a2, -a3, -a4], x * coscf)
    uimag = signal.lfilter([1, a1, a5], [1, -a1, -a2, -a3, -a4], x * sincf)
    envx = gain * np.sqrt(ureal * ureal + uimag * uimag + 1e-8)

    tptBW = BWy * tpt * ERB * 1.019
    a = np.exp(-tptBW)
    a1 = 4.0 * a
    a2 = -6.0 * a * a
    a3 = 4.0 * a * a * a
    a4 = -a * a * a * a
    a5 = 4.0 * a * a
    gain = 2.0 * (1 - a1 - a2 - a3 - a4) / (1 + a1 + a5)

    ureal = signal.lfilter([1, a1, a5], [1, -a1, -a2, -a3, -a4], y * coscf)
    uimag = signal.lfilter([1, a1, a5], [1, -a1, -a2, -a3, -a4], y * sincf)

    envy = gain * np.sqrt(ureal * ureal + uimag * uimag + 1e-8)

    return envx, envy


def eb_BWadjust(control, BWmin, BWmax, Level1):
    cRMS = np.sqrt(np.mean(control**2))
    cdB = 20 * np.log10(cRMS) + Level1
    if cdB < 50:
        BW = BWmin
    elif cdB > 100:
        BW = BWmax
    else:
        BW = BWmin + ((cdB - 50) / 50) * (BWmax - BWmin)
    return BW


def eb_GammatoneBM2_origin(x, BWx, y, BWy, fs, cf):
    earQ = 9.26449
    minBW = 24.7
    ERB = minBW + (cf / earQ)
    nx = len(x)
    ny = len(y)
    nsamp = min(nx, ny)
    x = x[:nsamp]
    y = y[:nsamp]

    tpt = 2 * math.pi / fs
    tptBW = BWx * tpt * ERB * 1.019
    a = np.exp(-tptBW)
    a1 = 4.0 * a
    a2 = -6.0 * a * a
    a3 = 4.0 * a * a * a
    a4 = -a * a * a * a
    a5 = 4.0 * a * a
    gain = 2.0 * (1 - a1 - a2 - a3 - a4) / (1 + a1 + a5)

    npts = len(x)
    cn = np.cos(tpt * cf)
    sn = np.sin(tpt * cf)
    coscf = np.zeros(npts)
    sincf = np.zeros(npts)
    cold = 1
    sold = 0
    coscf[0] = cold
    sincf[0] = sold

    for n in range(1, npts):
        arg = cold * cn + sold * sn
        sold = sold * cn - cold * sn
        cold = arg
        coscf[n] = cold
        sincf[n] = sold

    coscf = np.ones(npts, dtype="float32")
    sincf = np.ones(npts, dtype="float32")

    ureal = signal.lfilter([1, a1, a5], [1, -a1, -a2, -a3, -a4], x * coscf)
    uimag = signal.lfilter([1, a1, a5], [1, -a1, -a2, -a3, -a4], x * sincf)

    BMx = gain * (ureal * coscf + uimag * sincf)
    envx = gain * np.sqrt(ureal * ureal + uimag * uimag)

    tptBW = BWy * tpt * ERB * 1.019
    a = np.exp(-tptBW)
    a1 = 4.0 * a
    a2 = -6.0 * a * a
    a3 = 4.0 * a * a * a
    a4 = -a * a * a * a
    a5 = 4.0 * a * a
    gain = 2.0 * (1 - a1 - a2 - a3 - a4) / (1 + a1 + a5)

    ureal = signal.lfilter([1, a1, a5], [1, -a1, -a2, -a3, -a4], y * coscf)
    uimag = signal.lfilter([1, a1, a5], [1, -a1, -a2, -a3, -a4], y * sincf)

    BMy = gain * (ureal * coscf + uimag * sincf)
    envy = gain * np.sqrt(ureal * ureal + uimag * uimag)

    return envx, BMx, envy, BMy


def eb_GammatoneBM2(x, BWx, y, BWy, fs, cf, coscf, sincf):
    earQ = 9.26449
    minBW = 24.7
    ERB = minBW + (cf / earQ)
    nx = len(x)
    ny = len(y)
    nsamp = min(nx, ny)
    x = x[:nsamp]
    y = y[:nsamp]

    tpt = 2 * math.pi / fs
    tptBW = BWx * tpt * ERB * 1.019
    a = np.exp(-tptBW)
    a1 = 4.0 * a
    a2 = -6.0 * a * a
    a3 = 4.0 * a * a * a
    a4 = -a * a * a * a
    a5 = 4.0 * a * a
    gain = 2.0 * (1 - a1 - a2 - a3 - a4) / (1 + a1 + a5)

    # npts = len(x)
    # cn = np.cos(tpt * cf)
    # sn = np.sin(tpt * cf)
    # coscf = np.zeros(npts)
    # sincf = np.zeros(npts)
    # cold = 1
    # sold = 0
    # coscf[0] = cold
    # sincf[0] = sold
    #
    # for n in range(1, npts):
    #     arg = cold * cn + sold * sn
    #     sold = sold * cn - cold * sn
    #     cold = arg
    #     coscf[n] = cold
    #     sincf[n] = sold

    # coscf = np.ones(npts, dtype="float32")
    # sincf = np.ones(npts, dtype="float32")

    ureal = signal.lfilter([1, a1, a5], [1, -a1, -a2, -a3, -a4], x * coscf)
    uimag = signal.lfilter([1, a1, a5], [1, -a1, -a2, -a3, -a4], x * sincf)

    BMx = gain * (ureal * coscf + uimag * sincf)
    envx = gain * np.sqrt(ureal * ureal + uimag * uimag)

    tptBW = BWy * tpt * ERB * 1.019
    a = np.exp(-tptBW)
    a1 = 4.0 * a
    a2 = -6.0 * a * a
    a3 = 4.0 * a * a * a
    a4 = -a * a * a * a
    a5 = 4.0 * a * a
    gain = 2.0 * (1 - a1 - a2 - a3 - a4) / (1 + a1 + a5)

    ureal = signal.lfilter([1, a1, a5], [1, -a1, -a2, -a3, -a4], y * coscf)
    uimag = signal.lfilter([1, a1, a5], [1, -a1, -a2, -a3, -a4], y * sincf)

    BMy = gain * (ureal * coscf + uimag * sincf)
    envy = gain * np.sqrt(ureal * ureal + uimag * uimag)

    return envx, BMx, envy, BMy


def eb_EnvCompressBM(envsig, bm, control, attnOHC, thrLow, CR, fsamp, Level1):
    thrHigh = 100.0
    small = 1.0e-30
    logenv = np.maximum(control, small)
    logenv = logenv / 5e-4
    logenv = Level1 + 20 * np.log10(logenv)
    logenv = np.minimum(logenv, thrHigh)
    logenv = np.maximum(logenv, thrLow)

    gain = -attnOHC - (logenv - thrLow) * (1 - (1 / CR))
    gain = 10 ** (gain / 20)
    flp = 800

    bLP, aLP = signal.butter(1, flp / (0.5 * fsamp), "lowpass")
    gain = signal.lfilter(bLP, aLP, gain)
    y = gain * envsig
    b = gain * bm
    return y, b


def eb_EnvAlign(x, y):
    fsamp = 24000
    range = 100
    lags = round(0.001 * range * fsamp)
    npts = len(x)
    lags = min(lags, npts)

    xy_full = np.correlate(x, y, "full")
    mid = int((len(xy_full) - 1) / 2)
    xy = xy_full[(mid - (lags - 1)) : (mid + lags)]
    location = np.argmax(np.abs(xy))
    delay = lags - 1 - location
    if delay > 0:
        pad = np.zeros(delay)
        y = np.concatenate((y[delay:npts], pad), axis=0)
    else:
        delay = int(-delay)
        pad = np.zeros(delay)
        y = np.concatenate((pad, y[: npts - delay]), axis=0)

    return y


def eb_EnvSL2(env, bm, attnIHC, Level1):
    small = 1.0e-10
    env1 = env + small
    env1 = env1 / 5e-4
    y = Level1 - attnIHC + 20 * np.log10(env1)
    y = np.maximum(y, 0.0)
    gain = (y + small) / (env + small)
    b = gain * bm
    return y, b


def eb_EnvSL(env, attnIHC, Level1):
    small = 1.0e-30
    y = Level1 - attnIHC + 20 * np.log10(env + small)
    y = np.maximum(y, 0.0)
    return y


def eb_IHCadapt(xdB, xBM, delta, fsamp):
    dsmall = 1.0001
    if delta < dsmall:
        delta = dsmall
    tau1 = 2
    tau2 = 60
    tau1 = 0.001 * tau1
    tau2 = 0.001 * tau2
    T = 1 / fsamp
    R1 = 1 / delta
    R2 = 0.5 * (1 - R1)
    R3 = R2
    C1 = tau1 * (R1 + R2) / (R1 * R2)
    C2 = tau2 / ((R1 + R2) * R3)

    a11 = R1 + R2 + R1 * R2 * (C1 / T)
    a12 = -R1
    a21 = -R3
    a22 = R2 + R3 + R2 * R3 * (C2 / T)
    denom = 1.0 / (a11 * a22 - a21 * a12)
    R1inv = 1.0 / R1
    R12C1 = R1 * R2 * (C1 / T)
    R23C2 = R2 * R3 * (C2 / T)
    nsamp = len(xdB)
    gain = np.ones(len(xdB))
    ydB = np.zeros(len(xdB))
    V1 = 0.0
    V2 = 0.0
    small = 1.0e-30

    # for n in range(nsamp):
    #     V0 = xdB[n]
    #     b1 = V0 * R2 + R12C1 * V1
    #     b2 = R23C2 * V2
    #     V1 = denom * (a22 * b1 - a12 * b2)
    #     V2 = denom * (-a21 * b1 + a11 * b2)
    #     out = (V0 - V1) * R1inv
    #     out = np.maximum(out, 0.0)
    #     ydB[n] = out
    #     gain[n] = (out + small) / (V0 + small)
    # yBM = gain * xBM

    ydB = xdB
    yBM = xBM

    return ydB, yBM


def eb_IHCadapt1(xdB, delta, fsamp):
    dsmall = 1.0001
    if delta < dsmall:
        delta = dsmall
    tau1 = 2
    tau2 = 60
    tau1 = 0.001 * tau1
    tau2 = 0.001 * tau2
    T = 1 / fsamp
    R1 = 1 / delta
    R2 = 0.5 * (1 - R1)
    R3 = R2
    C1 = tau1 * (R1 + R2) / (R1 * R2)
    C2 = tau2 / ((R1 + R2) * R3)

    a11 = R1 + R2 + R1 * R2 * (C1 / T)
    a12 = -R1
    a21 = -R3
    a22 = R2 + R3 + R2 * R3 * (C2 / T)
    denom = 1.0 / (a11 * a22 - a21 * a12)

    R1inv = 1.0 / R1
    R12C1 = R1 * R2 * (C1 / T)
    R23C2 = R2 * R3 * (C2 / T)

    nsamp = len(xdB)
    ydB = np.zeros(len(xdB))
    V1 = 0.0
    V2 = 0.0

    for n in range(nsamp):
        V0 = xdB[n]
        b1 = V0 * R2 + R12C1 * V1
        b2 = R23C2 * V2
        V1 = denom * (a22 * b1 - a12 * b2)
        V2 = denom * (-a21 * b1 + a11 * b2)
        out = (V0 - V1) * R1inv
        out = np.maximum(out, 0.0)
        ydB[n] = out
    return ydB


def eb_BMaddnoise(x, thr, Level1):
    gn = 10 ** ((thr - Level1) / 20.0)
    noise = gn * np.random.randn(len(x))
    y = x + noise
    return y


def eb_GroupDelayComp(xenv, BW, cfreq, fsamp):
    nchan = len(BW)
    earQ = 9.26449
    minBW = 24.7
    ERB = minBW + (cfreq / earQ)
    tpt = 2 * math.pi / fsamp
    tptBW = tpt * 1.019 * BW * ERB
    a = np.exp(-tptBW)
    a1 = 4.0 * a
    a2 = -6.0 * a * a
    a3 = 4.0 * a * a * a
    a4 = -a * a * a * a
    a5 = 4.0 * a * a

    gd = np.zeros(nchan)
    for n in range(nchan):
        w, gd[n] = signal.group_delay(([1, a1[n], a5[n]], [1, -a1[n], -a2[n], -a3[n], -a4[n]]), 1)
    gmin = min(gd)
    gd = gd - gmin
    gmax = max(gd)
    correct = gmax - gd

    yenv = np.zeros_like(xenv)

    for n in range(nchan):
        r = xenv[n, :]
        npts = len(r)
        pad = np.zeros(correct[n])
        r_new = np.concatenate((pad, r[: npts - correct[n]]), axis=0)
        yenv[n, :] = r_new
    return yenv


def eb_aveSL(env, control, attnOHC, thrLow, CR, attnIHC, Level1):
    thrHigh = 100.0
    small = 1.0e-30
    logenv = np.maximum(control, small)
    logenv = Level1 + 20 * np.log10(logenv)
    logenv = np.minimum(logenv, thrHigh)
    logenv = np.maximum(logenv, thrLow)
    gain = -attnOHC - (logenv - thrLow) * (1 - (1.0 / CR))
    logenv = np.maximum(env, small)
    logenv = Level1 + 20 * np.log10(logenv)
    logenv = np.maximum(logenv, 0)
    xdB = logenv + gain - attnIHC
    xdB = np.maximum(xdB, 0.0)
    return xdB


if __name__ == "__main__":
    cf = eb_CenterFreq(32)

    cf1 = eb_CenterFreq(nchan=32, shift=0.02)

    HL = [80, 85, 90, 80, 90, 80]

    attnOHCx, BWminx, lowkneex, CRx, attnIHCx = eb_LossParameters(HL, cf)

    print(cf.shape)
