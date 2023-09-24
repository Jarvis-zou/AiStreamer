import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src/NeMo'))
from src.streamer import AiStreamer
from src.ui import MainWindow
from PyQt5.QtWidgets import QApplication

# openai_key = os.environ.get('OPENAI_API_KEY')
openai_key = 'your personal key'

# set configuration
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo", help="Model type you want to choose")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--room_id", type=int, default=5050)
parser.add_argument("--streamer", type=str, default='fengge',
                    help="Streamer name, should be same with instructions file name")
parser.add_argument("--hifigan", type=str, default=r"/home/zoujiawei/ckpt/hifigan.ckpt", help="Dir where hifigan model stores")
parser.add_argument("--fs2", type=str, default=r"/home/zoujiawei/ckpt/fs2.ckpt", help="Dir where fastpitch2 model stores")
parser.add_argument("--wav2lip_model", type=str, default=r"C:\Users\ZouJiawei\Desktop\Advanced_explore\source\checkpoints\wav2lip_gan.pth", help="Dir where wav2lip model stores")
parser.add_argument("--text_input", type=str, default=r".\GPT_Inputs", help="Dir path where streamer.jsonl stores")
parser.add_argument("--audio_input", type=str, default=r".\Audio_Inputs",
                    help="Dir path where streamer/streamer.mp3 stores")
parser.add_argument("--audio_output", type=str, default=r".\Audio_Outputs\fengge",
                    help="Dir path where streamer/streamer.wav will be stored")
parser.add_argument("--video_source", type=str, default=r"C:\Users\ZouJiawei\Desktop\Advanced_explore\source",
                    help="Path to speaker video file")

args = parser.parse_args()

if __name__ == '__main__':
    streamer = AiStreamer(openai_key, args)  # 创建streamer对象
    input_text = '对啊，好像我还能没看到上面中国主播啊或者网红啊去过对马岛，我可能也算是第一个去对马岛的了吧，到时候深度解密一下对马岛到底什么样儿啊，也没去过对马岛第一次去挺紧张的'
    streamer.generate_audio(input_text)
    # app = QApplication(sys.argv)
    # main_window = MainWindow(streamer=streamer)  # 启动主窗口，同时预载各种参数
    # main_window.show()
    # sys.exit(app.exec_())
