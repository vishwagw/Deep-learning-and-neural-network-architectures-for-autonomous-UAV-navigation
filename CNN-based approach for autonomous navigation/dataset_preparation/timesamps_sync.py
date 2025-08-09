import pandas as pd
from moviepy.editor import VideoFileClip

# Extract video timestamps
clip = VideoFileClip("flight1.mp4")
video_timestamps = [i/clip.fps for i in range(int(clip.fps*clip.duration))]

# Align with telemetry
telemetry = pd.read_csv('telemetry.csv')
aligned_data = []
for ts in video_timestamps:
    closest_idx = (telemetry['timestamp'] - ts*1000).abs().argsort()[0]
    aligned_data.append(telemetry.iloc[closest_idx])