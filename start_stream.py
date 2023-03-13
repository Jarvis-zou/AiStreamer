import argparse
import os
from src.streamer import AiStreamer

openai_key = os.environ.get('OPENAI_API_KEY')

# set configuration
parser = argparse.ArgumentParser()
parser.add_argument("-model_name", type=str, default="gpt-3.5-turbo", help="Model type you want to choose")
parser.add_argument("-streamer", type=str, default='fengge', help="Streamer name, should be same with instructions file name")
parser.add_argument("-tts_models", type=str, default="./tts_models", help="Dir where tts models stores")
parser.add_argument("--text_input", type=str, default="./GPT_Inputs", help="Dir path where streamer.jsonl stores")
parser.add_argument("--audio_input", type=str, default="./Audio_Inputs", help="Dir path where streamer/streamer.mp3 stores")
parser.add_argument("--audio_output", type=str, default="./Audio_Outputs", help="Dir path where streamer/streamer.wav will be stored")
parser.add_argument("--voice", type=str, default=r"C:\Users\ZouJiawei\Desktop\Advanced_explore\AiStreamer\examples\wav\template.wav", help="Path to speaker voice file")

args = parser.parse_args()


streamer = AiStreamer(openai_key, args)
streamer.generate_audio()