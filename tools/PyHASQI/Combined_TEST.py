import soundfile
import numpy as np
from eb_EarModel import *
from eb_EnvSmooth import eb_EnvSmooth
from eb_melcor import eb_melcor
from eb_SpectDiff import eb_SpectDiff
from eb_BMcovary import eb_BMcovary
from eb_AveCovary2 import eb_AveCovary2
import time
from pesq import pesq
# clean_path = "F:\\Hearing_Loss\\matlab\\HASQI_Fig6_Final\\HASQI\\fileid_132_clean.wav"
# noisy_path = "F:\\Hearing_Loss\\matlab\\HASQI_Fig6_Final\\HASQI\\fileid_132_noisy.wav"
clean_path = "F:\\Hearing_Loss\\HL_compen_codes\\PyHASQI\\TEST_wavs\\clean_fig6_fileid_100.wav"
noisy_path = "F:\\Hearing_Loss\\HL_compen_codes\\PyHASQI\\TEST_wavs\\noisy_fig6_fileid_100.wav"


(clean_audio, fs1) = soundfile.read(clean_path)
(noisy_audio, fs2) = soundfile.read(noisy_path)


start = time.time()
temp_PESQ = pesq(fs1, clean_audio, noisy_audio, 'wb')
end = time.time()
print(end-start)

HL = [80, 85, 90, 80, 90, 80]

Level1 = 65
eq = 2

# noisy_fig6 = Fig6_Amplification(HL, clean_audio, fs2)

clean_audio = np.array(clean_audio, dtype="float32")
noisy_audio = np.array(noisy_audio, dtype="float32")

start = time.time()
xenv, xBM, yenv, yBM, xSL, ySL, fsamp = eb_EarModel_compute(clean_audio, fs1, noisy_audio, fs2, HL, eq, Level1)
end = time.time()

print(end-start)

# np.savez('xenv_new.npz', xenv_name=xenv)
#
# np.savez('yenv_new.npz', yenv_name=yenv)
#
# np.savez('xBM.npz', xBM_name=xBM)
#
# np.savez('yBM.npz', yBM_name=yBM)
#
# np.savez('xSL.npz', xSL_name=xSL)
#
# np.savez('ySL.npz', ySL_name=ySL)

# data_xenv = np.load('xenv.npz')
# xenv = data_xenv['xenv_name']  # 数据格式依旧是numpy.array
#
# data_yenv = np.load('yenv.npz')
# yenv = data_yenv['yenv_name']  # 数据格式依旧是numpy.array



segsize = 16

xdB = eb_EnvSmooth(xenv, segsize, fsamp)
ydB = eb_EnvSmooth(yenv, segsize, fsamp)

thr = 2.5
addnoise = 0.0

CepCorr, xy, vad = eb_melcor(xdB, ydB, thr, addnoise)

if CepCorr == 0:
    Combined = 0
    Nonlin = 0
    Linear = 0
    raw = np.zeros((1,4))

dloud, dnorm, dslope = eb_SpectDiff(xSL, ySL)

segcov = 16

start = time.time()
sigcov, sigMSx, sigMSy = eb_BMcovary(xBM, yBM, segcov, fsamp)
end = time.time()

print(end-start)

thr = 2.5

avecov, syncov = eb_AveCovary2(sigcov, sigMSx, thr)

BMsync5 = syncov[4]
if BMsync5 == 0:
    Combined = 0
    Nonlin = 0
    Linear = 0
    raw = np.zeros(4)
# d = 0
d = dloud[1]
d = d/2.5
d = 1.0 - d
d = np.minimum(d, 1)
d = np.maximum(d, 0)
Dloud = d

# d = 0
d = dslope[1]
d = 1.0 - d
d = np.minimum(d, 1)
d = np.maximum(d, 0)
Dslope = d

Nonlin = (CepCorr ** 2) * BMsync5
Linear = 0.579*Dloud + 0.421*Dslope

Combined = Nonlin*Linear
raw = [Combined, Nonlin, Linear]

print(raw)



