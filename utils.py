import openai
import os
import jsonlines


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

