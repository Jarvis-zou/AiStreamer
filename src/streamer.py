import multiprocessing as mp
import threading
import queue
from contextlib import ExitStack
import openai
import asyncio
import jsonlines
import random
import soundfile as sf
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM
from src.listen_chats import listen
from src.Wav2Lip.inference import sync_lip, load_model
from src.NeMo.nemo.collections.tts.models import HifiGanModel, FastPitchModel



class AiStreamer:
    def __init__(self, api_key, args):
        self.api_key = api_key
        self.args = args
        self.wav2lip_model = None
        self.frontend = None
        self.am_predictor = None
        self.voc_predictor = None
        self.content_id = 0  # 当前音视频文件生成后的名称（编号），根据编号进行播放

        # 初始化预设video路径
        self.not_talking_videos_source_path = os.path.join(os.path.join(self.args.video_source, self.args.streamer),
                                                           'not_talking_source')
        self.talking_videos_source_path = os.path.join(os.path.join(self.args.video_source, self.args.streamer),
                                                       'talking_source')
        self.sync_result_path = os.path.join(os.path.join(self.args.video_source, self.args.streamer),
                                             'sync_result')

        # 建立输入的通信队列
        self.normal_chats = mp.Queue()  # 普通弹幕队列
        self.sc_chats = mp.Queue()  # sc留言队列
        self.questions = mp.Queue()  # 结果队列,格式为(question: result_avi path})

    def listen_chats(self):
        """实时监听直播间的普通弹幕和付费SC留言"""
        asyncio.run(listen(self.args.room_id, self.normal_chats, self.sc_chats))

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
        # openai.api_key = self.api_key
        # # 区分3.5模型和其他老模型，因为接口调用方式不一样
        # if self.args.model_name == 'gpt-3.5-turbo':
        #     massages = self.read_jsonl()
        #     result = openai.ChatCompletion.create(
        #         model=self.args.model_name,
        #         messages=massages,
        #         temperature=0.4,
        #         top_p=1,
        #         n=1,
        #         presence_penalty=0.2,
        #         frequency_penalty=0.2
        #     )
        #     answer = result['choices'][0]['message']['content']
        #     print(f'回答：{answer}')
        answer = "对啊，好像我还能没看到上面中国主播啊或者网红啊去过对马岛，我可能也算是第一个去对马岛的了吧，到时候深度解密一下对马岛到底什么样儿啊，也没去过对马岛第一次去挺紧张的"
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

    def generate_audio(self, input_text):
        """paddlespeech本地推理"""
        output_dir = Path(self.args.audio_output)
        output_dir.mkdir(parents=True, exist_ok=True)

        merge_sentences = True
        fs = 24000
        am_output_data = get_am_output(input=input_text,
                                       am_predictor=self.am_predictor,
                                       am="fastspeech2_mix",
                                       frontend=self.frontend,
                                       lang="mix",
                                       merge_sentences=merge_sentences,
                                       speaker_dict=os.path.join(self.args.encoder, "phone_id_map.txt"),
                                       spk_id=0)
        wav = get_voc_output(voc_predictor=self.voc_predictor, input=am_output_data)

        # 保存文件
        wav_id = str(self.content_id) + ".wav"
        sf.write(output_dir / wav_id, wav, samplerate=fs)
        print(f'TTS数据生成完毕...')

    def generate_video(self):
        # 随机挑选一个视频进行合成
        talking_videos_source_list = self.load_video(self.talking_videos_source_path)
        face_index = random.randrange(len(talking_videos_source_list))
        face = talking_videos_source_list[face_index]
        sync_lip(ckpt=self.wav2lip_model,
                 face=face,
                 audio_input=os.path.join(self.args.audio_output, str(self.content_id) + '.wav'),
                 outfile=os.path.join(self.sync_result_path, str(self.content_id) + '.avi'))
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

    def generate_answer(self, question):
        """从Inputs路径下的对应jsonl文件中读取用户的提问, 根据不同模型调用不同API并返回答案"""
        # 修改问题格式
        uname, msg = question[0], question[1]
        processed_question = uname + '问：' + msg

        # 生成回答
        self.content_id += 1  # 确认编号
        self.get_inputs_from_typing(question=processed_question)  # 将问题写入jsonl中的message模板
        text_answer = self.generate_text()
        self.generate_audio(input_text=text_answer)
        self.generate_video()
        self.questions.put((processed_question, os.path.join(self.sync_result_path, str(self.content_id) + '.avi')))  # 队列存入回答

    def processing_chats(self):
        """预加载模型并监听输入队列"""
        # 预加载lip_sync model
        if self.wav2lip_model is None:
            self.wav2lip_model = load_model(self.args.wav2lip_model)

        # 预加载TTS模型
        tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
        bert_model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-roberta-wwm-ext-large").to(self.args.device).eval()
        vocoder_model_pt = HifiGanModel.load_from_checkpoint(checkpoint_path=self.args.hifigan).to(self.args.device).eval()

        # 主循环
        while True:
            if not self.sc_chats.empty():  # 优先读取sc弹幕的留言
                sc_question = self.sc_chats.get()
                print(f'SC留言:{sc_question}')
                self.generate_answer(question=sc_question)
            if not self.normal_chats.empty() and self.sc_chats.empty():  # 如果sc留言队列为空则开始处理普通弹幕
                normal_question = self.normal_chats.get()
                print(f'普通留言:{normal_question}')
                self.generate_answer(question=normal_question)

    def start_stream(self):
        """使用进程启动各个模块对消息列表进行监听"""
        # 启动 listen_chats 进程
        listen_chats_process = mp.Process(target=self.listen_chats)
        listen_chats_process.start()

        # 启动 processing_chats 进程
        processing_chats_process = mp.Process(target=self.processing_chats)
        processing_chats_process.start()

