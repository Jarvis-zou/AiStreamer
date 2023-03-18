from moviepy.editor import *

# class Tool:
#     def __init__(self):


def clip_video(video_path, hms_start, hms_end, save_path):
    """
    :param video_path: str类型，待剪辑的原视频文件路径
    :param hms_start: tuple/list类型，剪辑起始时间(hour, minute, second)
    :param hms_end: tuple/list类型，剪辑结束时间(hour, minute, second)
    :param save_path: str类型，剪辑结果保存路径
    """
    try:
        clip = VideoFileClip(video_path).subclip(hms_start, hms_end)
        clip.write_videofile(save_path)
    except TypeError:
        print(f'WARNING: Wrong from Form of start param should be (hour, minute, second) or (minute, second) or second(int)')
    except OSError:
        pass

def clip_audio(audio_path, hms_start, hms_end, save_path):
    clip = AudioFileClip(audio_path).subclip(hms_start, hms_end)
    clip.write_audiofile(save_path)

