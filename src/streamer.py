import multiprocessing as mp
import openai
import os
import jsonlines
import random
from pathlib import Path
from src.TTS.models.synthesizer.inference import Synthesizer
from src.TTS.models.encoder import inference as encoder
from src.TTS.models.vocoder.hifigan import inference as gan_vocoder
from src.Wav2Lip.inference import sync_lip, load_model
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
                                             'sync_result/result_video.avi')

        # 预加载TTS模型路径

        # 预加载lip_sync model
        self.wav2lip_model = load_model(self.args.wav2lip_model)

        # 建立通信队列和signal
        self.audio_ready_signal = False  # 确定TTS语音是否已经生成完成
        self.video_ready_signal = False  # 确定lip视频是否已经生成完成
        self.text_input = mp.Queue()  # 问题输入队列

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
            print(f'回答：{answer}')
            return answer
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
            print(f'回答：{answer}')
            return answer

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
            if self.audio_ready_signal is True:
                wav_file_path = Path(self.args.audio_output) / Path(self.args.streamer + '.wav')
                audio = AudioSegment.from_file(wav_file_path)
                play(audio)
                self.audio_ready_signal = False

    def generate_audio(self, input_text):
        """根据GPT API的文字答案生成对应人物音色的wav语音文件并保存到指定的路径下"""
        tts = gTTS(text=input_text, lang='zh-cn')
        # 保存WAV音频到指定文件路径
        file_name = self.args.streamer + '.wav'
        save_path = os.path.join(self.args.audio_output, file_name)
        tts.save(save_path)
        self.audio_ready_signal = True
        print(f'TTS数据生成完毕...')

    def generata_video(self):
        # 随机挑选一个视频进行合成
        talking_videos_source_list = self.load_video(self.talking_videos_source_path)
        face_index = random.randrange(len(talking_videos_source_list))
        face = talking_videos_source_list[face_index]
        sync_lip(ckpt=self.wav2lip_model,
                 face=face,
                 audio_input=os.path.join(self.args.audio_output, self.args.streamer + '.wav'),
                 outfile=self.sync_result_path)
        self.video_ready_signal = True
        print(f'video数据生成完毕...')

    @staticmethod
    def load_video(video_path):
        """创建进程队列保存VideoCapture对象"""
        video_list = []
        for file_name in os.listdir(video_path):
            video = os.path.join(video_path, file_name)
            video_list.append(video)  # 先把所有文件的cap对象都读取到列表内
        return video_list

    def load_next_video(self):
        """给出主窗口下一个待播放的视频地址"""
        not_talking_video_list = self.load_video(self.not_talking_videos_source_path)
        next_video_index = random.randrange(len(not_talking_video_list))
        next_video = not_talking_video_list[next_video_index]
        return next_video

    def generate_answer(self):
        """从Inputs路径下的对应jsonl文件中读取用户的提问, 根据不同模型调用不同API并返回答案"""
        while True:
            if not self.text_input.empty():  # 监听到文本输入则进入pipline: 答案生成 -> 语音合成 -> 口型生成
                text_answer = self.generate_text()
                self.generate_audio(input_text=text_answer)
                self.generata_video()

    def start_stream(self):
        """使用进程启动各个模块对消息列表进行监听"""
        # 启动 generate_video 进程
        load_next_video_process = mp.Process(target=self.load_next_video)
        load_next_video_process.start()

        # 启动generate_answer 进程
        generate_answer_process = mp.Process(target=self.generate_answer)
        generate_answer_process.start()
