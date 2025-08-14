import soundfile
import numpy as np
import librosa
import os
from HASQI_revised import HASQI_v2
import xlsxwriter as xw

def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)

    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs

def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

def gen_HASQI_fea():
    filename = 'F:\\Hearing_Loss\\DATASET\\HL_fig6_DSDATASET_Limit\\TEST_SET\\HASQI.txt'
    HASQI_dict = dict()
    for line in open(filename):
        # 将读取的每行的结果类型转为一个列表
        line_split = line.replace('\n', '')
        list_HASQI = line_split.split(",")
        # list_HASQI = line.split()
        HASQI_dict[list_HASQI[0]] = float(list_HASQI[1])
    return HASQI_dict

def gen_ht_fea():
    filename_ht = 'F:\\Hearing_Loss\\DATASET\\HL_fig6_DSDATASET_Limit\\TEST_SET\\ht.txt'
    ht_dict = dict()

    for line in open(filename_ht):
        # 将读取的每行的结果类型转为一个列表
        # list_ht = line.split()
        line_split = line.replace('\n', '')
        list_ht = line_split.split(",")
        ht_values = []
        for index_ht in range(1, len(list_ht)):
            ht_values.append(float(list_ht[index_ht]))
        ht_dict[list_ht[0]] = ht_values

    return ht_dict

speech_dir = "F:\\Hearing_Loss\\DATASET\\HL_fig6_DSDATASET_Limit\\TEST_SET\\limit_noisy_testset_16k"
clean_dir = "F:\\Hearing_Loss\\DATASET\\HL_fig6_DSDATASET_Limit\\TEST_SET\\limit_clean_testset_16k"
workspace = "F:\\Hearing_Loss\\DATASET\\HL_fig6_DSDATASET_Limit\\TEST_SET"
HASQI_dict = gen_HASQI_fea()
ht_dict = gen_ht_fea()
speech_names = []

Level1 = 65
eq = 2

fileName = "F:\\Hearing_Loss\\DATASET\\HL_fig6_DSDATASET_Limit\\TEST_SET\\test_HASQI_compare.xlsx"

workbook = xw.Workbook(fileName)

worksheet1 = workbook.add_worksheet("sheet1")  # 创建子表
worksheet1.activate()  # 激活表
title = ['测试语音', '原始HASQI', 'Py简化版HASQI']  # 设置表头
worksheet1.write_row('A1', title)  # 从A1单元格开始写入表头

xls_index = 2

for dirpath, dirnames, filenames in os.walk(speech_dir):
    for filename in filenames:
         if filename.lower().endswith(".wav"):
            speech_names.append(os.path.join(dirpath, filename))

for speech_na in speech_names:
    speech_na_basename = os.path.basename(speech_na)
    speech_fpart = os.path.splitext(speech_na_basename)[0]
    name = os.path.splitext(os.path.basename(speech_na))[0]
    mixture_compen_dir = "F:\\Hearing_Loss\\DATASET\\HL_fig6_DSDATASET_Limit\\TEST_SET\\limit_noisy_testset_16k_fig6"
    clean_compen_dir = "F:\\Hearing_Loss\\DATASET\\HL_fig6_DSDATASET_Limit\\TEST_SET\\limit_clean_testset_16k_fig6"
    mixture_compen_path = os.path.join(mixture_compen_dir, speech_na_basename)
    clean_compen_path = os.path.join(clean_compen_dir, speech_na_basename)

    mixture_compen, fs1 = soundfile.read(mixture_compen_path, dtype="float32")
    clean_compen, fs2 = soundfile.read(clean_compen_path, dtype="float32")

    wave_HASQI = HASQI_dict[name]
    wave_ht = ht_dict[name]

    cur_HASQI = HASQI_v2(clean_compen, fs1, mixture_compen, fs1, wave_ht, eq, Level1)

    insertData = [name, wave_HASQI, cur_HASQI]

    row = 'A' + str(xls_index)
    worksheet1.write_row(row, insertData)

    xls_index += 1

    print(name)

workbook.close()  # 关闭表





