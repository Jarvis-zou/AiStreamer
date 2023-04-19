import os
import json
import subprocess
import numpy as np
import soundfile as sf
import pandas as pd
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


def cut_audio(raw_path, save_path, sample_rate, seg_len):
    """
    对原始wav数据重采样到指定采样率和单声道16bit，并且将其重新切段并保存
    
    输入：
        raw_path: 原始的wav音频数据存放路径
        save_path: ffmpeg按照sample_rate和seg_len切分后的wav音频保存路径
        sample_rate: 希望将原始wav重采样到的目标采样率
        seg_len: 切分后的wav长度，单位秒
    """
    for file_name in os.listdir(raw_path):
        file_path = os.path.join(raw_path, file_name)
        save_name = save_path + '/' + file_name.split('.')[0]
        subprocess.run(f"ffmpeg -hide_banner -loglevel panic -i {file_path} -ac 1 -ar {sample_rate} -f s16le - " +
                       f"| ffmpeg -hide_banner -loglevel panic -f s16le -ar {sample_rate} -i - -f segment -segment_time {seg_len} {save_name}_%03d.wav",
                       shell=True, check=True)
        print(f'{file_name} processing done.')


def vad_preprocess(wav_dir, save_dir):
    """
    对切分好的固定长度的音频段使用VAD模型进行识别

    输入：
        wav_dir:ffmpeg按照seg_len长度切分的wav音频段保存路径
        save_dir: vad检测结果的保存路径，保存形式可参照modelscope文档 https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary
    """
    # 定义pipeline，因为用scp文件形式进行处理所以添加了output_dir选项
    inference_pipeline = pipeline(
        task=Tasks.voice_activity_detection,
        model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
        model_revision='v1.1.8',
        device=DEVICE,
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


def slice_wav(data_root, vad_output_dir, wav_dir, duration_range, save_dir):
    """
    根据VAD切分结果对wav片段做进一步切分，并保证音频段在固定长度区间内吗，处理结果将存入data_root下的meta.csv文件中
    
    输入：
        data_root: 该路径下保存所有的前处理中产生的中间数据
        vad_output_dir: 上一步中保存vad检测结果的路径
        wav_dir： 第一步中ffmpeg初次切分保存wav数据的路径
        duration_range： [最短时间， 最长时间], 单位秒，所有VAD检测后的数据经过切分应当都处于这个长度区间内
        save_dir: 切分并裁剪拼接后处于duration_range内的短音频数据的保存路径
    """
    # 读取VAD的切分结果
    vad_result_file = os.path.join(vad_output_dir, '1best_recog/text')
    with open(vad_result_file, 'r+') as vad_result:
        slices_info = vad_result.readlines()
    
    
    # 遍历slice起止点截取wav音频
    meta_file = os.path.join(data_root, 'meta.csv')
    print(f'开始数据切片')
    with open(meta_file, 'w') as meta:  # 信息写入meta.csv文件
        for info in slices_info:
            info = info[:-1]  # 去掉换行符
            wav_name, slice_nodes = info.split('.wav')[0], eval(info.split('.wav')[1])
            wav, sr = sf.read(os.path.join(wav_dir, wav_name + '.wav'))

            # 根据duration区间对wav进行裁剪拼接
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


def asr_detection(wav_dir, save_dir):
    """
    对切分好的固定长度的音频段使用VAD模型进行识别
    
    输入: 
        wav_dir: 上一步中切分好的段音频wav slices文件保存的路径
        save_dir: asr检测结果的保存路径，保存形式可参照modelscope文档 https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary
    """
    # 定义pipeline，因为用scp文件形式进行处理所以添加了output_dir选项
    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
        device=DEVICE,
        output_dir=save_dir)
    
    # 讲上一步切分好的wav片段写入scp文件中
    scp_file_path= os.path.join(save_dir, 'wavs.scp')
    with open(scp_file_path, 'w+', encoding='utf8') as scp:
        for file_name in os.listdir(wav_dir):
            file_path = os.path.join(wav_dir, file_name)
            line = file_name + '  ' + file_path + '\n'
            scp.write(line)
    print(f'scp文件生成完毕.')

    # # 调用ASR识别语音，PUNC识别标点
    print(f'开始进行ASR语音识别.')
    inference_pipeline(audio_in=scp_file_path)  
    print(f'ASR结果生成完毕.')


def punc_detection(save_dir):
    """
    对切分好的固定长度的音频段使用VAD模型进行识别
    
    输入: 
        wav_dir: 上一步中切分好的段音频wav slices文件保存的路径
        save_dir: asr检测结果的保存路径，保存形式可参照modelscope文档 https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary
    """
    # 定义pipeline，因为用scp文件形式进行处理所以添加了output_dir选项
    inference_pipeline = pipeline(
        task=Tasks.punctuation,
        model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
        model_revision="v1.1.7",
        device=DEVICE,
        output_dir=save_dir)
    
    # # 讲上一步切分好的wav片段写入scp文件中
    # scp_file_path= os.path.join(save_dir, 'wavs.scp')
    # with open(scp_file_path, 'w+', encoding='utf8') as scp:
    #     for file_name in os.listdir(wav_dir):
    #         file_path = os.path.join(wav_dir, file_name)
    #         line = file_name + '  ' + file_path + '\n'
    #         scp.write(line)
    # print(f'scp文件生成完毕.')

    # # 调用ASR识别语音，PUNC识别标点
    print(f'开始进行PUNC标点符号识别.')
    result = inference_pipeline(text_in='我们都是木头人不会讲话不会动')  
    print(f'PUNC结果生成完毕.{result}')


# 路径初始化, 首先在data_root路径下将这些路径创建好
data_root = r'/home/zoujiawei/data'
raw_wav_dir = os.path.join(data_root, 'raw')  # 原始wav音频存放路径
wav_cuts_save_dir = os.path.join(data_root, 'wavs')  # 5分钟切段wav存放路径
vad_save_dir = os.path.join(data_root, 'vad')  # vad语音起止点检测结果存放路径
slices_save_dir = os.path.join(data_root, 'wav_slices')  # 根据vad结果切分的wav切片存放路径
asr_save_dir = os.path.join(data_root, 'asr')  # asr识别结果的保存路径
punc_save_dir = os.path.join(data_root, 'punc')  # 标点符号识别结果的保存路径

# 预处理相关参初始化
DEVICE = 'cpu'
sample_rate = 16000  # 采样率设置，由于调用ASR模型进行语音识别时会采样到16000，所以切分数据时先采样至16000
seg_len = 300  # 切段长度，单位为秒，默认300秒(5分钟切段)，可修改
slice_duration = [2.0, 15.0]  # 音频切片的长度区间，切分后保证所有音频长度都在该区间内

# cut_audio(raw_wav_dir, wav_cuts_save_dir, sample_rate, seg_len)  # 先对原始数据进行第一次切分
# vad_preprocess(wav_cuts_save_dir, vad_save_dir)  # vad模型将每句话单独切分成音频
# slice_wav(data_root, vad_save_dir, wav_cuts_save_dir, slice_duration, slices_save_dir)  # 按照VAD结果将5分钟的音频片段切分成2-15秒的切片
# asr_detection(slices_save_dir, asr_save_dir)  # 调用ASR模型进行语音文本识别
punc_detection(punc_save_dir)  # 调用PUNC模型进行标点符号识别

