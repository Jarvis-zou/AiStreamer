import openai
import os
import jsonlines


class AiStreamer:
    def __init__(self, api_key, args):
        self.api_key = api_key
        self.args = args

    def read_jsonl(self):
        """
        读取jsonl文件内容
        :returns
            contents: [list]
        """
        jsonl_name = self.args.streamer + '.jsonl'
        jsonl_path = os.path.join(self.args.inputs_dir, jsonl_name)  # instruction文件的相对路径
        with open(jsonl_path, 'r+', encoding='utf8') as f:
            contents = []
            for items in jsonlines.Reader(f):
                contents.append(items)
        return contents

    def generate_text(self):
        """从Inputs路径下的对应jsonl文件中读取用户的提问, 根据不同模型调用不同API并返回答案"""
        openai.api_key = self.api_key

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

