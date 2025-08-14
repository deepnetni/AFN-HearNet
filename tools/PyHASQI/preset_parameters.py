import os
import numpy as np

from .eb_operations import *


def pre_compute_params(nchan, length, fs, cfreq):
    coscf_total = np.zeros((nchan, length))
    sincf_total = np.zeros((nchan, length))

    for chan in range(nchan):
        cf = cfreq[chan]

        tpt = 2 * math.pi / fs
        cn = np.cos(tpt * cf)
        sn = np.sin(tpt * cf)
        cold = 1
        sold = 0
        coscf_total[chan, 0] = cold
        sincf_total[chan, 0] = sold
        for n in range(1, length):
            arg = cold * cn + sold * sn
            sold = sold * cn - cold * sn
            cold = arg
            coscf_total[chan, n] = cold
            sincf_total[chan, n] = sold
    return coscf_total, sincf_total


def generate_filter_params(lenx):
    nchan = 12
    cfreq = eb_CenterFreq(nchan)
    shift = 0.02
    cfreq1 = eb_CenterFreq(nchan, shift)
    fsamp = 24000

    Env2_coscf_total, Env2_sincf_total = pre_compute_params(nchan, lenx, fsamp, cfreq1)
    BM2_coscf_total, BM2_sincf_total = pre_compute_params(nchan, lenx, fsamp, cfreq)

    base_dir = os.path.dirname(__file__)

    np.savez(f"{base_dir}/Env2_coscf.npz", Env2_coscf=Env2_coscf_total)
    np.savez(f"{base_dir}/Env2_sincf.npz", Env2_sincf=Env2_sincf_total)

    np.savez(f"{base_dir}/BM2_coscf.npz", BM2_coscf=BM2_coscf_total)
    np.savez(f"{base_dir}/BM2_sincf.npz", BM2_sincf=BM2_sincf_total)


if __name__ == "__main__":
    import soundfile

    nchan = 12
    cfreq = eb_CenterFreq(nchan)
    shift = 0.02
    cfreq1 = eb_CenterFreq(nchan, shift)

    # clean_path = "F:\\Hearing_Loss\\DATASET\\HL_fig6_DSDATASET_Limit\\TEST_SET\\limit_noisy_testset_16k\\p232_001.wav"
    # (clean_audio, fs1) = soundfile.read(clean_path)
    # x24, fsamp = eb_Resamp24kHz(clean_audio, fs1)
    fsamp = 24000
    # lenx = 71808
    # lenx = 144000
    # lenx = 120000
    lenx = 119808
    # lenx = 240000

    Env2_coscf_total, Env2_sincf_total = pre_compute_params(nchan, lenx, fsamp, cfreq1)
    BM2_coscf_total, BM2_sincf_total = pre_compute_params(nchan, lenx, fsamp, cfreq)

    np.savez("Env2_coscf.npz", Env2_coscf=Env2_coscf_total)
    np.savez("Env2_sincf.npz", Env2_sincf=Env2_sincf_total)

    np.savez("BM2_coscf.npz", BM2_coscf=BM2_coscf_total)
    np.savez("BM2_sincf.npz", BM2_sincf=BM2_sincf_total)
