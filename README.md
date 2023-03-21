# 如何本地使用

Created by: Jarvis Z
Created time: March 21, 2023 5:08 PM
Last edited by: Jarvis Z
Last edited time: March 21, 2023 6:00 PM

# 如何本地使用

本文将介绍如何在本地使用AiStreamer。

## 项目内容

AiStreamer提供了一套完整的框架，它能够实现对你喜欢的主播的快速克隆，并且设定AI主播们的个性，让他们制造节目效果等等。它为你提供了一个创造性的平台，让你可以用AI技术制作出有趣的内容。如果你想使用AiStreamer来打造自己的节目，本文将介绍如何在本地使用AiStreamer。请注意请不要使用本项目用于用于虚假新闻、网络欺诈、网络钓鱼等不良行为，此外，数字人克隆技术也可能被用于人身攻击、隐私侵犯等不良行为。

## 先决条件

在开始使用AiStreamer之前，您需要安装相应的库，请执行以下命令：

```
git clone https://github.com/Jarvis-zou/AiStreamer.git
pip install -r requirements.txt
sudo apt-get install ffmpeg
```

## 获取权重以及测试用例

在开始使用前，您需要下载TTS和Wav2Lip模块的预训练权重。您可以通过以下链接下载权重：

| Model | Description | Link to the model |
| --- | --- | --- |
| Wav2Lip | Highly accurate lip-sync | https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels%2Fwav2lip%2Epth&parent=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels&ga=1 |
| Wav2Lip + GAN | Slightly inferior lip-sync, but better visual quality | https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels%2Fwav2lip%5Fgan%2Epth&parent=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels&ga=1 |
| Expert Discriminator | Weights of the expert discriminator | https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels%2Flipsync%5Fexpert%2Epth&parent=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels&ga=1 |
| Visual Quality Discriminator | Weights of the visual disc trained in a GAN setup | https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels%2Fvisual%5Fquality%5Fdisc%2Epth&parent=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels&ga=1 |
| s3fd | pre-trained model | https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth |

将下载的Wav2Lip权重文件放入`path to your weights/`文件夹中，其中`s3fd.pth`应当被下载到`src/Wav2Lip/face_detection/detection/sfd/s3fd.pth`。

或者你也可以在[这里](https://pan.baidu.com/s/1oM0igqeGZ03e9XvFPnZQZA)下载所有的数据，提取码为rqp1，文件中包含了所有的模型和演示视频用例。

解压后文件目录为：

```
├── source/
│   ├── checkpoints/
│   │   ├── lipsync_expert.pth
│   │   ├── visual_quality_disc.pth
│   │   ├── wav2lip_gan.pth
│   │   └── wav2lip.pth
│   ├── not_talking_source/
│   │   ├── not_talking_1.avi
│   │   ├── not_talking_2.avi
│   │   ├── not_talking_3.avi
│   │   └── ...
│   ├── talking_source/
│   │   ├── talking_1.avi
│   │   ├── talking_2.avi
│   │   ├── talking_3.avi
│   │   └── ...
│   ├── sync_result/
│   ├── ckpt(TTS模型，由于现阶段TTS使用gtts库代替，所以这里的模型用不到)/
```

## 运行

要运行程序，需要先将openai api key设置为环境变量，然后请执行以下命令：

```
python start_stream.py --streamer <ai_name> --video_source <path to source>
```

这将启动程序，您将会看到一个QT界面，您可以在其中询问你的问题。

## 反馈

如果您在使用AiStreamer时遇到任何问题，请随时联系我们，我们将竭诚为您服务。