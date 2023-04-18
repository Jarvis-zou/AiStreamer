import os
import json
import subprocess
import numpy as np
import soundfile as sf
import pandas as pd
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


def cut_audio(dir_path, save_path, sample_rate, seg_len):
    """对原始wav数据重采样到指定采样率和单声道16bit，并且将其重新切段并保存"""
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        save_name = save_path + '/' + file_name.split('.')[0]
        subprocess.run(f"ffmpeg -hide_banner -loglevel panic -i {file_path} -ac 1 -ar {sample_rate} -f s16le - " +
                       f"| ffmpeg -hide_banner -loglevel panic -f s16le -ar {sample_rate} -i - -f segment -segment_time {seg_len} {save_name}_%03d.wav",
                       shell=True, check=True)
        print(f'{file_name} processing done.')


def slice_wav(vad_output_dir, wav_dir, duration_range, save_dir):
    """根据VAD切分结果对wav片段做进一步切分，保证音频段在固定长度区间内"""
    # 读取VAD的切分结果
    vad_result_file = os.path.join(vad_output_dir, '1best_recog/text')
    with open(vad_result_file, 'r+') as vad_result:
        slices_info = vad_result.readlines()
    
    # 根据duration区间对wav进行裁剪拼接
    meta_file = os.path.join(save_dir, 'meta.csv')
    print(f'开始数据切片')
    with open(meta_file, 'w') as meta:  # 信息写入meta.csv文件
        for info in slices_info:
            info = info[:-1]  # 去掉换行符
            wav_name, slice_nodes = info.split('.wav')[0], eval(info.split('.wav')[1])
            wav, sr = sf.read(os.path.join(wav_dir, wav_name + '.wav'))

            # 遍历slice起止点截取wav音频
            count = 0
            for i, (start, end) in enumerate(slice_nodes):
                    segment_samples = []
                    segment_samples.append([int(start / 1000 * sr), int(end / 1000 * sr)])
                    if (end - start) / 1000 < duration_range[0]:
                        continue
                    _wavs = [np.concatenate([wav[seg[0]:seg[1]] for seg in segment_samples])]
                    if (_wavs[0].shape[0] / sr) > duration_range[1]:
                        _k = 2 ** np.ceil(np.log2((_wavs[0].shape[0]) / sr / duration_range[1]))
                        for i in range(int(_k)):
                            _wavs.append(_wavs[0][int(i/_k*_wavs[0].shape[0]) : int((i+1)/_k*_wavs[0].shape[0])])
                        _wavs = _wavs[1:]
                    for _wav in _wavs:
                        save_path = os.path.join(save_dir, f"{wav_name}_{str(count).zfill(4)}.wav")
                        sf.write(save_path, _wav, sr)
                        meta_info = {'audio_filepath': save_path, "duration": round(_wav.shape[0] / sr, 3)}
                        meta.writelines(json.dumps(meta_info) + '\n')
                        count += 1 
    
    # 统计结果
    with open(meta_file) as meta:
        meta = [json.loads(i) for i in meta.readlines()]
        meta = pd.DataFrame(meta)
    total_audio_hour = round(meta['duration'].sum() / 60 / 60, 2)
    print(f'切片完成，总计{len(meta)}条数据，总计{total_audio_hour}小时数据')

def vad_preprocess(wav_dir, save_dir):
    """对切分好的固定长度的音频段使用VAD模型进行识别"""
    # 定义pipeline，因为用scp文件形式进行处理所以添加了output_dir选项
    inference_pipeline = pipeline(
        task=Tasks.voice_activity_detection,
        model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
        model_revision='v1.1.8',
        device='cpu',
        output_dir = save_dir)
    
    # 讲上一步切分好的wav片段写入scp文件中
    scp_file_path= os.path.join(save_dir, 'wavs.scp')
    with open(scp_file_path, 'w+', encoding='utf8') as scp:
        for file_name in os.listdir(wav_dir):
            file_path = os.path.join(wav_dir, file_name)
            line = file_name + '  ' + file_path + '\n'
            scp.write(line)
    print(f'scp文件生成完毕.')

    # VAD模型对wav切片进行进一步细分，细分结果为起止时间点，单位ms
    inference_pipeline(audio_in=scp_file_path)  
    print(f'VAD结果生成完毕.')

    # 读取VAD切分结果，将wav重新切分


def asr_punc_detection(wav_dir, save_dir):
    """对切分好的固定长度的音频段使用VAD模型进行识别"""
    # 定义pipeline，因为用scp文件形式进行处理所以添加了output_dir选项
    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
        punc_model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
        device='cpu',
        output_dir = save_dir)
    
    # 讲上一步切分好的wav片段写入scp文件中
    scp_file_path= os.path.join(save_dir, 'wavs.scp')
    with open(scp_file_path, 'w+', encoding='utf8') as scp:
        for file_name in os.listdir(wav_dir):
            file_path = os.path.join(wav_dir, file_name)
            line = file_name + '  ' + file_path + '\n'
            scp.write(line)
    print(f'scp文件生成完毕.')

    # 调用ASR识别语音，PUNC识别标点
    inference_pipeline(audio_in=scp_file_path)  
    print(f'ASR PUNC结果生成完毕.')


# 路径初始化, 首先在data_root路径下将这些路径创建好
data_root = r'/home/zoujiawei/data'
raw_wav_dir = os.path.join(data_root, 'raw')
wav_cuts_save_dir = os.path.join(data_root, 'wavs')
vad_save_dir = os.path.join(data_root, 'vad')
slices_save_dir = os.path.join(data_root, 'wav_slices')

# 预处理相关参初始化
sample_rate = 16000  # 采样率设置，由于调用ASR模型进行语音识别时会采样到16000，所以切分数据时先采样至16000
seg_len = 300  # 切段长度，单位为秒，默认300秒(5分钟切段)，可修改
slice_duration = [2.0, 15.0]  # 音频切片的长度区间，切分后保证所有音频长度都在该区间内

# cut_audio(raw_wav_dir, wav_cuts_save_dir, sample_rate, seg_len)  # 先对原始数据进行第一次切分
# vad_preprocess(wav_cuts_save_dir, vad_save_dir)  # vad模型将每句话单独切分成音频
# slice_wav(vad_save_dir, wav_cuts_save_dir, slice_duration, slices_save_dir)  # 按照VAD结果将5分钟的音频片段切分成2-15秒的切片
vad_asr_punc_detection(wav_cuts_save_dir, vad_save_dir)

