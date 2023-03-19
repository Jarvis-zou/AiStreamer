import argparse
import os
import sys
from src.streamer import AiStreamer
from src.ui import MainWindow
from PyQt5.QtWidgets import QApplication

# openai_key = os.environ.get('OPENAI_API_KEY')
openai_key = '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'

# set configuration
parser = argparse.ArgumentParser()
parser.add_argument("-model_name", type=str, default="gpt-3.5-turbo", help="Model type you want to choose")
parser.add_argument("-streamer", type=str, default='fengge',
                    help="Streamer name, should be same with instructions file name")
parser.add_argument("-tts_models", type=str, default=r"C:\Users\Administrator\Desktop\workspace\source\ckpt", help="Dir where tts models stores")
parser.add_argument("-wav2lip_model", type=str, default=r"C:\Users\Administrator\Desktop\workspace\source\checkpoints\wav2lip_gan.pth", help="Dir where wav2lip models stores")
parser.add_argument("--text_input", type=str, default="./GPT_Inputs", help="Dir path where streamer.jsonl stores")
parser.add_argument("--audio_input", type=str, default="./Audio_Inputs",
                    help="Dir path where streamer/streamer.mp3 stores")
parser.add_argument("--audio_output", type=str, default="./Audio_Outputs",
                    help="Dir path where streamer/streamer.wav will be stored")
parser.add_argument("--voice", type=str,
                    default=r"C:\Users\Administrator\Desktop\workspace\AiStreamer\examples\wav\fengge016.wav",
                    help="Path to speaker voice file")
parser.add_argument("--video_source", type=str, default=r"C:\Users\Administrator\Desktop\workspace\source",
                    help="Path to speaker video file")

args = parser.parse_args()

if __name__ == '__main__':
    streamer = AiStreamer(openai_key, args)  # 创建streamer对象
    app = QApplication(sys.argv)
    main_window = MainWindow(streamer=streamer)  # 启动主窗口，同时预载各种参数
    main_window.show()
    sys.exit(app.exec_())
