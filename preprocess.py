from utils.tools import clip_video

video_path = 'C:/Users/ZouJiawei/Desktop/Advanced_explore/MockingBird/audio_data/fengge_data.mp4'
save_path = 'fengge_pianduan.mp4'
clip_video(video_path, (0, 0, 50), (0, 1, 5), save_path)