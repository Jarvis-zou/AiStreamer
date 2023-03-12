import argparse
import os
from utils import AiStreamer

openai_key = os.environ.get('OPENAI_API_KEY')
openai_key = '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'

# set configuration
parser = argparse.ArgumentParser()
parser.add_argument("-model_name", type=str, default="gpt-3.5-turbo", help="Model type you want to choose")
parser.add_argument("-streamer", type=str, help="Streamer name, should be same with instructions file name")
parser.add_argument("--text_dir", type=str, default="./GPT_Inputs", help="Dir path where streamer.jsonl stores")
parser.add_argument("--audio_dir", type=str, default="./Audio_Inputs", help="Dir path where streamer/streamer.mp3 stores")

args = parser.parse_args()

streamer = AiStreamer(openai_key, args)
result = streamer.get_inputs_from_typing(question='峰哥,你如何看待原神玩家?')