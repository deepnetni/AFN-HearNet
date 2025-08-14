import numpy as np
from .eb_operations import *
import copy


def eb_EarModel_compute_origin(x, xsamp, y, ysamp, HL, itype, Level1):
    nchan = 12
    mdelay = 0

    cfreq = eb_CenterFreq(nchan)

    attnOHCy, BWminy, lowkneey, CRy, attnIHCy = eb_LossParameters(HL, cfreq)

    if itype == 0:
        HLx = 0 * HL
    else:
        HLx = HL

    attnOHCx, BWminx, lowkneex, CRx, attnIHCx = eb_LossParameters(HLx, cfreq)
    HLmax = np.array([100, 100, 100, 100, 100, 100], dtype="float32")
    shift = 0.02

    cfreq1 = eb_CenterFreq(nchan, shift)
    _, BW1, _, _, _ = eb_LossParameters(HLmax, cfreq1)

    # if itype == 1:
    #     x = Fig6_Amplification(HL, x, xsamp)

    x24, _ = eb_Resamp24kHz(x, xsamp)

    y24, fsamp = eb_Resamp24kHz(y, ysamp)

    nxy = min(len(x24), len(y24))
    x24 = x24[:nxy]
    y24 = y24[:nxy]

    # x24, y24 = eb_InputAlign(x24, y24)

    nsamp = len(x24)

    xmid = eb_MiddleEar(x24, fsamp)
    ymid = eb_MiddleEar(y24, fsamp)

    xdB = np.zeros((nchan, nsamp), dtype="float32")
    ydB = np.zeros((nchan, nsamp), dtype="float32")
    xBM = np.zeros((nchan, nsamp), dtype="float32")
    yBM = np.zeros((nchan, nsamp), dtype="float32")

    xave = np.zeros(nchan, dtype="float32")
    yave = np.zeros(nchan, dtype="float32")
    xcave = np.zeros(nchan, dtype="float32")
    ycave = np.zeros(nchan, dtype="float32")
    BWx = np.zeros(nchan, dtype="float32")
    BWy = np.zeros(nchan, dtype="float32")

    for n in range(nchan):
        xcontrol, ycontrol = eb_GammatoneEnv2_origin(xmid, BW1[n], ymid, BW1[n], fsamp, cfreq1[n])

        BWx[n] = eb_BWadjust(xcontrol, BWminx[n], BW1[n], Level1)
        BWy[n] = eb_BWadjust(ycontrol, BWminy[n], BW1[n], Level1)

        xenv, xbm, yenv, ybm = eb_GammatoneBM2_origin(xmid, BWx[n], ymid, BWy[n], fsamp, cfreq[n])

        xave[n] = np.sqrt(np.mean(xenv**2))
        yave[n] = np.sqrt(np.mean(yenv**2))
        xcave[n] = np.sqrt(np.mean(xcontrol**2))
        ycave[n] = np.sqrt(np.mean(ycontrol**2))

        xc, xb = eb_EnvCompressBM(
            xenv, xbm, xcontrol, attnOHCx[n], lowkneex[n], CRx[n], fsamp, Level1
        )
        yc, yb = eb_EnvCompressBM(
            yenv, ybm, ycontrol, attnOHCy[n], lowkneey[n], CRy[n], fsamp, Level1
        )

        # yc = eb_EnvAlign(xc, yc)
        # yb = eb_EnvAlign(xb, yb)

        xc, xb = eb_EnvSL2(xc, xb, attnIHCx[n], Level1)
        yc, yb = eb_EnvSL2(yc, yb, attnIHCy[n], Level1)

        delta = 2.0
        xdB[n, :], xb = eb_IHCadapt(xc, xb, delta, fsamp)
        ydB[n, :], yb = eb_IHCadapt(yc, yb, delta, fsamp)

        IHCthr = -10.0

        xBM[n, :] = eb_BMaddnoise(xb, IHCthr, Level1)
        yBM[n, :] = eb_BMaddnoise(yb, IHCthr, Level1)

    if mdelay > 0:
        xdB = eb_GroupDelayComp(xdB, BWx, cfreq, fsamp)
        ydB = eb_GroupDelayComp(ydB, BWx, cfreq, fsamp)
        xBM = eb_GroupDelayComp(xBM, BWx, cfreq, fsamp)
        yBM = eb_GroupDelayComp(yBM, BWx, cfreq, fsamp)

    xSL = eb_aveSL(xave, xcave, attnOHCx, lowkneex, CRx, attnIHCx, Level1)
    ySL = eb_aveSL(yave, ycave, attnOHCy, lowkneey, CRy, attnIHCy, Level1)

    return xdB, xBM, ydB, yBM, xSL, ySL, fsamp
