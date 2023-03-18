import openai
import os
import jsonlines
import random
import numpy as np
import soundfile as sf
import multiprocess as mp
import queue
import cv2
from pathlib import Path
from src.TTS.models.synthesizer.inference import Synthesizer
from src.TTS.models.encoder import inference as encoder
from src.TTS.models.vocoder.hifigan import inference as gan_vocoder


class AiStreamer:
    def __init__(self, api_key, args):
        openai.api_key = api_key
        self.args = args

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
        jsonl_path = os.path.join(self.args.text_inputs, jsonl_name)  # jsonl文件的相对路径
        with open(jsonl_path, 'w+', encoding='utf8') as f:
            for content in contents:
                f.write(str(content).replace('\'', '\"') + '\n')

    def get_inputs_from_typing(self, question):
        """替代音频输入，将问题写入jsonl文件中"""
        self.write_jsonl(question)

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
            return result['choices'][0]['message']['content']
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
            return result['choices'][0]['message']['content']

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

    def generate_audio(self, input_text):
        """
        根据speaker音色和文本生成语音输出

        :param input_text: 待合成的文本
        :return:
            generate_wav: 根据文本和speaker音色合成的语音
            synthesizer.sample_rate: 采样频率
        """
        # load models
        encoder_path = self.find_model(Path(self.args.tts_models) / 'encoder')
        synthesizer_path = self.find_model(Path(self.args.tts_models) / 'synthesizer')
        vocoder_path = self.find_model(Path(self.args.tts_models) / 'vocoder')
        try:
            encoder.load_model(encoder_path)
            synthesizer = Synthesizer(synthesizer_path)
            vocoder = gan_vocoder
            vocoder.load_model(vocoder_path)
        except PermissionError:
            print(f'Cannot find tts models in {self.args.tts_models}')

        # preprocess wav file
        encoder_wav = synthesizer.load_preprocess_wav(self.args.voice)
        embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)  # generate voice embed
        embeds = [embed] * len(input_text)
        specs = synthesizer.synthesize_spectrograms(input_text, embeds)
        breaks = [spec.shape[1] for spec in specs]
        spec = np.concatenate(specs, axis=1)

        # generate voice
        generated_wav, output_sample_rate = vocoder.infer_waveform(spec)

        # adding breaks into voice
        b_ends = np.cumsum(np.array(breaks) * synthesizer.hparams.hop_size)
        b_starts = np.concatenate(([0], b_ends[:-1]))
        wavs = [generated_wav[start:end] for start, end in zip(b_starts, b_ends)]
        breaks = [np.zeros(int(0.15 * synthesizer.sample_rate))] * len(breaks)
        generated_wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])

        # erase blank in voice
        generated_wav = encoder.preprocess_wav(generated_wav)

        # adjustment
        generated_wav = generated_wav / np.abs(generated_wav).max() * 0.97
        file_name = self.args.streamer + '.wav'
        save_path = os.path.join(self.args.audio_output, file_name)
        sf.write(save_path, generated_wav, synthesizer.sample_rate)

        return generated_wav, synthesizer.sample_rate

    @staticmethod
    def load_video_queue(video_queue, video_path):
        """创建进程队列保存VideoCapture对象"""
        video_list = []
        for file_name in os.listdir(video_path):
            cap = cv2.VideoCapture(os.path.join(video_path, file_name))
            video_list.append(cap)  # 先把所有文件的cap对象都读取到列表内
            video_queue.put(cap)  # 初始化先把所有的视频加入队列

        return video_list

    #
    # @staticmethod
    # def read_video_queue(video_path):



    def generate_video_local_test(self):
        """生成视频"""
        not_talking_videos_source_path = os.path.join(os.path.join(self.args.video_source, self.args.streamer), 'not_talking_source')
        talking_videos_source_path = os.path.join(os.path.join(self.args.video_source, self.args.streamer), 'talking_source')
        sync_result_path = os.path.join(os.path.join(self.args.video_source, self.args.streamer), 'sync_result')

        video_queue = queue.Queue(maxsize=10)
        not_talking_video_list = self.load_video_queue(video_queue, not_talking_videos_source_path)
        # 播放当前队列的第一个视频
        cap = video_queue.get_nowait()
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imshow("frame", frame)
                cv2.waitKey(int(100 / fps))
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # cap.release()
                # print(f'{cap}released')
                next_video_index = random.randrange(len(not_talking_video_list))
                video_queue.put(not_talking_video_list[next_video_index])
                # cap = video_queue.get()
                cap = not_talking_video_list[0]



        # video_queue = mp.Queue(10)
        # self.load_video_queue(not_talking_videos_source_path)






    def start_stream(self):
        answer_text = self.generate_text()
        answer_audio, _ = self.generate_audio(answer_text)
