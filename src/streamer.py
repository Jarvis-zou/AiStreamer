import openai
import os
import jsonlines
import numpy as np
from TTS.models.synthesizer.inference import Synthesizer
from TTS.models.encoder import inference as encoder
from TTS.models.vocoder.hifigan import inference as gan_vocoder


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
        jsonl_path = os.path.join(self.args.text_dir, jsonl_name)  # jsonl文件的相对路径
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
        jsonl_path = os.path.join(self.args.text_dir, jsonl_name)  # jsonl文件的相对路径
        with open(jsonl_path, 'w+', encoding='utf8') as f:
            for content in contents:
                f.write(str(content).replace('\'', '\"') + '\n')

    def get_inputs_from_typing(self, question):
        """替代音频输入，将问题写入jsonl文件中"""
        self.write_jsonl(question)

    def transcribe_audio(self):
        """对指定音频进行语音识别"""
        audio_file_path = os.path.join(os.path.join(self.args.audio_dir, self.args.streamer), 'audio_input.m4a')
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

    def generate_audio(self, target_wav, save_path):
        """基于input_voice的声纹对input_text进行模拟，输出为使用克隆声音朗读input_text的音频wav"""
        encoder_wav = synthesizer.load_preprocess_wav(input_voice_path)  # preprocess wav file
        embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav,
                                                           return_partials=True)  # generate voice embed
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
        file_name = 'output.wav'
        sf.write(file_name, generated_wav, synthesizer.sample_rate)

        return generated_wav, synthesizer.sample_rate