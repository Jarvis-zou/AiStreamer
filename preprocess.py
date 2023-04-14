import whisper
import os

# 加载语音识别模型
model = whisper.load_model("medium")

# 音频文件夹和srt文件夹路径
audio_dir = r'C:\Users\ZouJiawei\Desktop\Advanced_explore\train_tts\wavs\fengge_video_2'
save_dir = r'C:\Users\ZouJiawei\Desktop\Advanced_explore\train_tts\labels'

# 遍历音频文件夹中的所有音频文件
for file_name in os.listdir(audio_dir):
    absolute_path = os.path.join(audio_dir, file_name)
    print(file_name)

    # 生成srt文件路径
    label_file_path = os.path.join(save_dir, 'list.txt')

    # 读取音频文件并识别出文本内容
    audio = whisper.load_audio(absolute_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    print(result.text)

    # 将识别结果写入srt文件中
    with open(label_file_path, 'a+', encoding='utf8') as label_file:
        content = 'wavs/' + file_name + '|' + result.text + '\n'
        label_file.write(content)