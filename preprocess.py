import os
import subprocess


def cut_audio(dir_path, save_path, sample_rate, seg_len):
    """对原始wav数据重采样到指定采样率和单声道16bit，并且将其重新切段并保存"""
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        save_name = save_path + '/' + file_name.split('.')[0]
        subprocess.run(f"ffmpeg -hide_banner -loglevel panic -i {file_path} -ac 1 -ar {sample_rate} -f s16le - " +
                       f"| ffmpeg -hide_banner -loglevel panic -f s16le -ar {sample_rate} -i - -f segment -segment_time {seg_len} {save_name}_%03d.wav",
                       shell=True, check=True)


raw_wav_dir = r'D:\workspace\audio_train\raw'
processed_wav_save_dir = r'D:\workspace\audio_train\wavs'
sample_rate = 22050  # 采样率设置
seg_len = 300  # 切段长度，单位为秒，默认300秒(5分钟切段)

cut_audio(raw_wav_dir, processed_wav_save_dir, sample_rate, seg_len)  # 先对原始数据进行切分
