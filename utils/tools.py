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
        start_time = hms_start[0] * 60 * 60 + hms_start[1] * 60 + hms_start[2]
        end_time = hms_end[0] * 60 * 60 + hms_end[1] * 60 + hms_end[2]
        clip = VideoClip(video_path).subclip(start_time, end_time)
        clip.write_video_file(save_path)
    except :
        print(f'Wrong from Form of start param should be (hour, minute, second)')