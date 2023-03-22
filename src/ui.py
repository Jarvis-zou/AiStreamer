import threading
import time
from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (
    QLabel, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QLineEdit, QPushButton, QPlainTextEdit, QDialog
)


class MainWindow(QMainWindow):
    def __init__(self, streamer):
        super().__init__()
        self.streamer = streamer
        self.streamer.start_stream()  # 初始化streamer，启动各个线程并提前加载模型

        # 是否要播放生成视频的信号
        self.answer_signal = False

        # 创建 QVideoWidget
        self.video_widget = QVideoWidget()
        self.video_widget.setFixedSize(720, 1200)
        # self.video_widget.setFixedSize(360, 600)

        # 设置窗口标题和大小
        self.setWindowTitle('Streaming')
        self.resize(1200, 900)

        # 创建媒体播放器并将其设置为 QVideoWidget
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_widget)

        # 加载并播放一个视频文件
        video_url = QUrl.fromLocalFile(self.streamer.load_next_video())
        self.media_player.setMedia(QMediaContent(video_url))
        self.media_player.stateChanged.connect(self.handle_state_changed)
        self.media_player.play()

        # 创建 QPushButton 用于提问
        self.text_ask_button = QPushButton("文本提问")
        self.text_ask_button.clicked.connect(self.ask_text_question)

        # 创建播放按钮
        self.play_button = QPushButton("看看回答")
        self.play_button.clicked.connect(lambda: self.play_specific_video())

        # 创建只读的QLabel组件
        self.text_label = QLabel()
        self.text_label.setAlignment(Qt.AlignCenter)
        self.text_label.setText("当前暂无问题")
        self.text_label.setStyleSheet("background-color: white; border: none;")

        # 布局
        video_container = QWidget()
        video_container.setFixedWidth(720)
        video_container.setLayout(QVBoxLayout())
        video_container.layout().addWidget(self.video_widget)

        text_container = QWidget()
        text_container.setLayout(QVBoxLayout())
        text_container.layout().addWidget(self.text_label)

        hbox = QHBoxLayout()
        hbox.addWidget(video_container)
        hbox.addWidget(text_container)

        container = QWidget()
        container.setLayout(hbox)
        self.setCentralWidget(container)

        # 创建 QPushButton 用于提问
        self.text_ask_button = QPushButton("文本提问")
        self.text_ask_button.clicked.connect(self.ask_text_question)

        # 将 QPushButton 添加到 QLabel 组件的布局中
        text_container.layout().addWidget(self.text_ask_button)
        text_container.layout().addWidget(self.play_button)

    def submit_text_question(self):
        question_text = self.question_input.toPlainText()
        self.text_label.setText(question_text)  # 展示当前问题
        self.streamer.get_inputs_from_typing(question_text)  # 将问题放入输入队列，开启处理流程
        self.text_input_dialog.close()

    def ask_text_question(self):
        # 创建一个 QDialog，设置标题和大小
        self.text_input_dialog = QDialog(self)
        self.text_input_dialog.setWindowTitle("Ask")
        self.text_input_dialog.resize(600, 600)

        # 更改新窗口的位置
        main_window_position = self.pos()
        new_dialog_x = main_window_position.x() + self.width()
        new_dialog_y = main_window_position.y()
        self.text_input_dialog.move(new_dialog_x, new_dialog_y)

        # 创建一个 QPlainTextEdit 用于文本输入
        self.question_input = QPlainTextEdit(self.text_input_dialog)
        self.question_input.setGeometry(10, 10, 590, 500)

        # 创建一个 QPushButton 提交问题
        self.submit_button = QPushButton("提交问题", self.text_input_dialog)
        self.submit_button.setGeometry(250, 550, 100, 30)
        self.submit_button.clicked.connect(self.submit_text_question)

        # 显示对话框
        self.text_input_dialog.show()

    def handle_state_changed(self, state):
        if state == QMediaPlayer.StoppedState:
            if self.answer_signal is False:
                new_video_url = QUrl.fromLocalFile(self.streamer.load_next_video())
                self.media_player.setMedia(QMediaContent(new_video_url))
                self.media_player.play()
            else:
                new_video_url = QUrl.fromLocalFile(self.streamer.sync_result_path)
                self.media_player.setMedia(QMediaContent(new_video_url))
                self.media_player.play()
                self.answer_signal = False

    def play_specific_video(self):
        self.answer_signal = True
        self.media_player.stop()





