from pathlib import Path
import soundfile as sf
import os
from paddlespeech.t2s.exps.syn_utils import get_am_output
from paddlespeech.t2s.exps.syn_utils import get_frontend
from paddlespeech.t2s.exps.syn_utils import get_predictor
from paddlespeech.t2s.exps.syn_utils import get_voc_output
import wave
import numpy as np

# 在其他环境中，记得修改下面这两个变量的路径
am_inference_dir = r"C:\Users\Administrator\Desktop\workspace\source\ckpt\fengge\fengge"
voc_inference_dir = r"C:\Users\Administrator\Desktop\workspace\source\ckpt\pwgan_aishell3_static_1.1.0\pwgan_aishell3_static_1.1.0"  # 这里以 pwgan_aishell3 为例子

# 音频生成的路径，修改成你音频想要保存的路径
wav_output_dir = "./Audio_Outputs"

# 选择设备[gpu / cpu]，这里以GPU为例子，
device = "gpu"

# 想要生成的文本和对应文件名

text_dict = {
    "1": "这原来是啥呀吗，就是属于那个熊呗，胖熊呗，熊现在想干啥想当猴了，这个属于啥呢，就是你原来这个体型是挺吃香的，你现在瘦下来之后吧，你没那个潜质，为什么呢，我跟你讲他一般都是什么样啊，就是那方面的呀，他得留那种小胡须，小胡须，它不能乱凌乱的胡须都不是，就留胡须吧，就必须得是哎萌萌的，像刚长一层草似的浅浅的，然后皮肤白白的，然后这边呢要从这个鬓角一直留下来，留到下面然后不能那么随意的流哎，，就必须得那个浅浅的一层，然后下面呢小胡子，然后上面的打理精修一下，然后这边呢带一个耳环你知道吧，然后头发呀一定要那种板寸，非常非常板正的板寸圆寸圆寸，然后穿要穿那种白色的纯白色的背心知道吗",
}

"""paddlespeech本地推理"""
# frontend
frontend = get_frontend(
    lang="mix",
    phones_dict=os.path.join(am_inference_dir, "phone_id_map.txt"),
    tones_dict=None
)

# am_predictor
am_predictor = get_predictor(
    model_dir=am_inference_dir,
    model_file="fastspeech2_mix" + ".pdmodel",
    params_file="fastspeech2_mix" + ".pdiparams",
    device=device)

# voc_predictor
voc_predictor = get_predictor(
    model_dir=voc_inference_dir,
    model_file="pwgan_aishell3" + ".pdmodel",    # 这里以 pwgan_aishell3 为例子，其它模型记得修改此处模型名称
    params_file="pwgan_aishell3" + ".pdiparams",
    device=device)

output_dir = Path(wav_output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

sentences = list(text_dict.items())

merge_sentences = True
fs = 24000
for utt_id, sentence in sentences:
    am_output_data = get_am_output(
        input=sentence,
        am_predictor=am_predictor,
        am="fastspeech2_mix",
        frontend=frontend,
        lang="mix",
        merge_sentences=merge_sentences,
        speaker_dict=os.path.join(am_inference_dir, "phone_id_map.txt"),
        spk_id=0, )
    wav = get_voc_output(
            voc_predictor=voc_predictor, input=am_output_data)
    # 保存文件
    sf.write(output_dir / (utt_id + ".wav"), wav, samplerate=fs)
