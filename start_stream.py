import argparse
import os
from utils import AiStreamer

openai_key = os.environ.get('OPENAI_API_KEY')

# set configuration
parser = argparse.ArgumentParser()
parser.add_argument("-model_name", type=str, help="Model type you want to choose")
parser.add_argument("-streamer", type=str, help="Streamer name, should be same with instructions file name")
parser.add_argument("--inputs_dir", type=str, default="./GPT_Inputs", help="Instructions dir path where instruction streamer.jsonl stores")

args = parser.parse_args()

streamer = AiStreamer(openai_key, args)
streamer.generate_text()