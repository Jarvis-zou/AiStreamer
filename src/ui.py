from PyQt5.QtCore import QUrl, QTimer, Qt
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (
    QLabel, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QLineEdit, QPushButton, QPlainTextEdit, QDialog
)
import os


class MainWindow(QMainWindow):
    def __init__(self, streamer):
        super().__init__()
        self.streamer = streamer
        self.streamer.start_stream()  # 初始化streamer，启动各个线程并提前加载模型
        self.generated_video = None

        # 是否要播放生成视频的信号
        self.answer_signal = False

        # 创建 QVideoWidget
        self.video_widget = QVideoWidget()
        # self.video_widget.setFixedSize(720, 1200)
        self.video_widget.setFixedSize(360, 600)

        # 设置窗口标题和大小
        self.setWindowTitle('Streaming')
        # self.resize(1200, 900)
        self.resize(600, 450)

        # 创建媒体播放器并将其设置为 QVideoWidget
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_widget)

        # 加载并播放一个视频文件
        video_url = QUrl.fromLocalFile(self.streamer.load_next_video())
        self.media_player.setMedia(QMediaContent(video_url))
        self.media_player.stateChanged.connect(self.handle_state_changed)
        self.media_player.play()

        # 创建看看回答按钮
        self.check_button = QPushButton("看看问题")
        self.check_button.clicked.connect(lambda: self.check_question())

        # 创建跳过按钮
        self.skip_button = QPushButton("跳过问题")
        self.skip_button.clicked.connect(lambda: self.skip_answer())

        # 创建播放按钮
        self.play_button = QPushButton("播放")
        self.play_button.clicked.connect(lambda: self.play_answer())

        # 显示问题的文本框
        self.answer_box = QPlainTextEdit()
        self.answer_box.setReadOnly(True)
        self.answer_box.setPlainText('请载入问题...')

        # 显示当前可播放回答列表
        self.avi_list = QPlainTextEdit()
        self.avi_list.setReadOnly(True)

        # 布局
        video_container = QWidget()
        video_container.setFixedWidth(720)
        video_container.setLayout(QVBoxLayout())
        video_container.layout().addWidget(self.video_widget)

        text_container = QWidget()
        text_container.setLayout(QVBoxLayout())

        hbox = QHBoxLayout()
        hbox.addWidget(video_container)
        hbox.addWidget(text_container)

        container = QWidget()
        container.setLayout(hbox)
        self.setCentralWidget(container)

        # 添加组件到布局
        text_container.layout().addWidget(self.avi_list)
        text_container.layout().addWidget(self.answer_box)
        text_container.layout().addWidget(self.check_button)
        text_container.layout().addWidget(self.skip_button)
        text_container.layout().addWidget(self.play_button)

        # 创建一个定时器，以每 5 秒更新 AVI 文件列表
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_avi_list)
        self.timer.start(1000)

    def check_question(self):
        # question_info = self.streamer.answers.get()
        # question, result = question_info[0], question_info[1]
        question, result = 'FOXDIE问：卵盘？几个T', 'C:\\Users\\ZouJiawei\\Desktop\\Advanced_explore\\source\\fengge\\sync_result\\1.avi'
        self.answer_box.clear()
        self.answer_box.setPlainText(f'当前问题：{question}')
        self.generated_video = result

    def skip_answer(self):
        if self.generated_video is not None:
            if os.path.isfile(self.generated_video):
                os.remove(self.generated_video)
                print(f"{self.generated_video} 已被删除")
            self.generated_video = None
        else:
            print(f"文件已被删除...")

    def play_answer(self):
        self.answer_signal = True
        self.media_player.stop()

    def update_avi_list(self):
        # 获取指定目录下所有的 AVI 文件名称
        avi_files = [f for f in os.listdir(self.streamer.sync_result_path) if f.endswith('.avi')]

        # 将 AVI 文件列表显示在文本编辑框中
        self.avi_list.clear()
        self.avi_list.appendPlainText('AVI 文件列表：')
        for avi_file in avi_files:
            self.avi_list.appendPlainText(avi_file)

    def handle_state_changed(self, state):
        if state == QMediaPlayer.StoppedState:
            if self.answer_signal is False or self.generated_video is None:
                new_video_url = QUrl.fromLocalFile(self.streamer.load_next_video())
                self.media_player.setMedia(QMediaContent(new_video_url))
                self.media_player.play()
                if self.generated_video is not None:
                    self.skip_answer()  # 播放完删除video
                    self.generated_video = None
            else:
                new_video_url = QUrl.fromLocalFile(self.generated_video)
                self.media_player.setMedia(QMediaContent(new_video_url))
                self.media_player.play()
                self.answer_signal = False







