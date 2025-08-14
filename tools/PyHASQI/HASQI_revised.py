import sys

# sys.path.append(".")

import soundfile
import numpy as np
from .eb_EarModel import *
from .eb_EnvSmooth import eb_EnvSmooth
from .eb_melcor import eb_melcor
from .eb_SpectDiff import eb_SpectDiff
from .eb_BMcovary import eb_BMcovary, eb_BMcovary_new
from .eb_AveCovary2 import eb_AveCovary2
from .eb_EarModel_origin import eb_EarModel_compute_origin


def HASQI_v2(x, fx, y, fy, HL, eq=2, Level1=65):
    """
    x is the fig5 compensated audio, y is the output of the compensation model.
    """
    xenv, xBM, yenv, yBM, xSL, ySL, fsamp = eb_EarModel_compute(x, fx, y, fy, HL, eq, Level1)
    segsize = 16
    xdB = eb_EnvSmooth(xenv, segsize, fsamp)
    ydB = eb_EnvSmooth(yenv, segsize, fsamp)
    thr = 0.5
    addnoise = 0.0
    CepCorr, xy, vad = eb_melcor(xdB, ydB, thr, addnoise)
    if CepCorr == 0:
        Combined = 0
        Nonlin = 0
        Linear = 0
        raw = np.zeros((1, 4))
        return Combined
    dloud, dnorm, dslope = eb_SpectDiff(xSL, ySL)
    segcov = 16
    sigcov, sigMSx, sigMSy = eb_BMcovary_new(xBM, yBM, segcov, fsamp)
    avecov, syncov = eb_AveCovary2(sigcov, sigMSx, thr)
    BMsync5 = syncov[4]
    if BMsync5 == 0:
        Combined = 0
        Nonlin = 0
        Linear = 0
        raw = np.zeros(4)
        return Combined
    d = dloud[1]
    d = d / 2.5
    d = 1.0 - d
    d = np.minimum(d, 1)
    d = np.maximum(d, 0)
    Dloud = d
    d = dslope[1]
    d = 1.0 - d
    d = np.minimum(d, 1)
    d = np.maximum(d, 0)
    Dslope = d

    Nonlin = (CepCorr**2) * BMsync5
    Linear = 0.579 * Dloud + 0.421 * Dslope

    Combined = Nonlin * Linear

    return Combined


def HASQI_v2_for_unfixedLen(x, fx, y, fy, HL, eq=2, Level1=65):
    xenv, xBM, yenv, yBM, xSL, ySL, fsamp = eb_EarModel_compute_origin(x, fx, y, fy, HL, eq, Level1)
    segsize = 16
    xdB = eb_EnvSmooth(xenv, segsize, fsamp)
    ydB = eb_EnvSmooth(yenv, segsize, fsamp)
    thr = 0.5
    addnoise = 0.0
    CepCorr, xy, vad = eb_melcor(xdB, ydB, thr, addnoise)
    if CepCorr == 0:
        Combined = 0
        Nonlin = 0
        Linear = 0
        raw = np.zeros((1, 4))
        return Combined
    dloud, dnorm, dslope = eb_SpectDiff(xSL, ySL)
    segcov = 16
    sigcov, sigMSx, sigMSy = eb_BMcovary_new(xBM, yBM, segcov, fsamp)
    avecov, syncov = eb_AveCovary2(sigcov, sigMSx, thr)
    BMsync5 = syncov[4]
    if BMsync5 == 0:
        Combined = 0
        Nonlin = 0
        Linear = 0
        raw = np.zeros(4)
        return Combined
    d = dloud[1]
    d = d / 2.5
    d = 1.0 - d
    d = np.minimum(d, 1)
    d = np.maximum(d, 0)
    Dloud = d
    d = dslope[1]
    d = 1.0 - d
    d = np.minimum(d, 1)
    d = np.maximum(d, 0)
    Dslope = d

    Nonlin = (CepCorr**2) * BMsync5
    Linear = 0.579 * Dloud + 0.421 * Dslope

    Combined = Nonlin * Linear

    return Combined


if __name__ == "__main__":
    clean_path = __file__.rsplit("/", 1)[0] + "/TEST_wavs/clean_fig6_fileid_100.wav"
    noisy_path = __file__.rsplit("/", 1)[0] + "/TEST_wavs/noisy_fig6_fileid_100.wav"

    (clean_audio, fs1) = soundfile.read(clean_path)
    (noisy_audio, fs2) = soundfile.read(noisy_path)

    HL = [80, 85, 90, 80, 90, 80]

    Level1 = 65
    eq = 2

    temp_HASQI = HASQI_v2(clean_audio, fs1, noisy_audio, fs2, HL, eq, Level1)
    print(temp_HASQI)
    temp_HASQI_ = HASQI_v2_for_unfixedLen(clean_audio, fs1, noisy_audio, fs2, HL, eq, Level1)
    print(temp_HASQI_)
