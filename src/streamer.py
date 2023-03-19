import threading
import queue
import openai
import os
import jsonlines
import random
from pathlib import Path
from src.TTS.models.synthesizer.inference import Synthesizer
from src.TTS.models.encoder import inference as encoder
from src.TTS.models.vocoder.hifigan import inference as gan_vocoder
from src.Wav2Lip.inference import sync_lip
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play


class AiStreamer:
    def __init__(self, api_key, args):
        self.api_key = api_key
        self.args = args

        # 初始化预设video路径
        self.not_talking_videos_source_path = os.path.join(os.path.join(self.args.video_source, self.args.streamer),
                                                           'not_talking_source')
        self.talking_videos_source_path = os.path.join(os.path.join(self.args.video_source, self.args.streamer),
                                                       'talking_source')
        self.sync_result_path = os.path.join(os.path.join(self.args.video_source, self.args.streamer),
                                             'sync_result/result_voice.mp4')

        # 初始化TTS模型路径

        # 初始化lip_sync model路径

        # 建立通信队列和signal
        self.play_audio_signal = False  # 确定TTS语音播放时机的signal
        self.lip_generation_finished = False  # 确定lip视频是否已经生成完成
        self.text_input = queue.Queue()  # 问题输入队列
        self.text_answer_queue = queue.Queue()  # API返回结果队列
        self.audio_answer_queue = queue.Queue()  # answer的TTS生成结果队列
        self.video_queue = queue.Queue()  # 存放下一个播放视频的队列

    def read_jsonl(self):
        """
        读取jsonl文件内容
        :returns
            contents: [list]
        """
        jsonl_name = self.args.streamer + '.jsonl'
        jsonl_path = os.path.join(self.args.text_input, jsonl_name)  # jsonl文件的相对路径
        with open(jsonl_path, 'r+', encoding='utf8') as f:
            contents = []
            for items in jsonlines.Reader(f):
                contents.append(items)
        return contents

    def write_jsonl(self, question):
        """写入jsonl文件"""
        contents = self.read_jsonl()
        contents[-1]["content"] = question  # 写入新问题
        jsonl_name = self.args.streamer + '.jsonl'
        jsonl_path = os.path.join(self.args.text_input, jsonl_name)  # jsonl文件的相对路径
        with open(jsonl_path, 'w+', encoding='utf8') as f:
            for content in contents:
                f.write(str(content).replace('\'', '\"') + '\n')

    def get_inputs_from_typing(self, question):
        """替代音频输入，将问题写入jsonl文件中"""
        self.write_jsonl(question)
        self.text_input.put(question)

    def transcribe_audio(self):
        """对指定音频进行语音识别"""
        audio_file_path = os.path.join(os.path.join(self.args.audio_input, self.args.streamer), 'audio_input.m4a')
        audio_file = open(audio_file_path, 'rb')
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        sentences = transcript['text'].split(',')
        sentences[0] = self.args.streamer  # 因为识别不准所以统一替换成指定的streamer名字
        transcript = ','.join(sentences)
        return transcript

    def generate_text(self):
        """从Inputs路径下的对应jsonl文件中读取用户的提问, 根据不同模型调用不同API并返回答案"""
        openai.api_key = self.api_key
        while True:
            if not self.text_input.empty():
                question = self.text_input.get()
                print(f'提问：{question}')
                # 区分3.5模型和其他老模型，因为接口调用方式不一样
                if self.args.model_name == 'gpt-3.5-turbo':
                    massages = self.read_jsonl()
                    result = openai.ChatCompletion.create(
                        model=self.args.model_name,
                        messages=massages,
                        temperature=0.8,
                        top_p=1,
                        n=1,
                        presence_penalty=0.2,
                        frequency_penalty=0.2
                    )
                    answer = result['choices'][0]['message']['content']
                    self.text_answer_queue.put(answer)
                    print(f'回答：{answer}')
                else:
                    prompt = self.read_jsonl()[0]  # 非3.5模型使用prompt而非messages形式作为输入，仅有一行信息
                    result = openai.Completion.create(
                        model=self.args.model_name,
                        prompt=prompt,
                        temperature=0.8,
                        top_p=1,
                        n=1,
                        presence_penalty=0.2,
                        frequency_penalty=0.2
                    )
                    answer = result['choices'][0]['message']['content']
                    self.text_answer_queue.put(answer)
                    print(f'回答：{answer}')

    @ staticmethod
    def find_model(model_dir):
        """寻找model_dir下后缀在check_list的文件的文件"""
        path = ""
        check_list = ['pt', 'pth']
        for file in os.listdir(model_dir):
            suffix = file.split('.')[-1]
            if suffix in check_list:
                path = model_dir / file
                print(path)
        return path

    def play_generated_audio(self):
        while True:
            if self.play_audio_signal:
                wav_file_path = Path(self.args.audio_output) / Path(self.args.streamer + '.wav')
                audio = AudioSegment.from_file(wav_file_path)
                play(audio)
                self.play_audio_signal = False  # 播放完成将signal关闭

    def generate_audio(self):
        while True:
            if not self.text_answer_queue.empty():
                input_text = self.text_answer_queue.get()
                tts = gTTS(text=input_text, lang='zh-cn')
                # 保存WAV音频到指定文件路径
                file_name = self.args.streamer + '.wav'
                save_path = os.path.join(self.args.audio_output, file_name)
                tts.save(save_path)
                print(f'TTS数据生成完毕...')
                self.audio_answer_queue.put(tts)

    @staticmethod
    def load_video(video_path):
        """创建进程队列保存VideoCapture对象"""
        video_list = []
        for file_name in os.listdir(video_path):
            video = os.path.join(video_path, file_name)
            video_list.append(video)  # 先把所有文件的cap对象都读取到列表内
        return video_list

    def generate_video(self):
        """给出主窗口下一个待播放的视频地址"""
        while True:
            if self.video_queue.empty():
                if not self.audio_answer_queue.empty():
                    # 随机挑选一个视频进行合成
                    talking_videos_source_list = self.load_video(self.talking_videos_source_path)
                    face_index = random.randrange(len(talking_videos_source_list))
                    face = talking_videos_source_list[face_index]
                    sync_lip(ckpt=self.args.wav2lip_model,
                             face=face,
                             audio=os.path.join(self.args.audio_output, self.args.streamer + '.wav'),
                             outfile=self.sync_result_path)
                    next_video = self.sync_result_path
                    self.video_queue.put(next_video)  # 下一个待播放视频为唇形生成好的视频
                    self.lip_generation_finished = True
                else:
                    not_talking_video_list = self.load_video(self.not_talking_videos_source_path)
                    next_video_index = random.randrange(len(not_talking_video_list))
                    next_video = not_talking_video_list[next_video_index]
                    self.video_queue.put(next_video)  # 下一个待播放视频为随机挑选的not talking video

    def start_stream(self):
        """使用进程启动各个模块对消息列表进行监听"""
        # 启动 generate_text 线程，监听self.text_input
        generate_text_thread = threading.Thread(target=self.generate_text)
        generate_text_thread.start()

        # 启动 generated_audio 线程，监听self.text_answer_queue
        generate_audio_thread = threading.Thread(target=self.generate_audio)
        generate_audio_thread.start()

        # 启动 generated_audio 线程，监听self.play_audio_signal
        play_generated_audio_thread = threading.Thread(target=self.play_generated_audio)
        play_generated_audio_thread.start()

        # 启动 generate_video 线程，监听self.audio_answer_queue
        generate_video_thread = threading.Thread(target=self.generate_video)
        generate_video_thread.start()
