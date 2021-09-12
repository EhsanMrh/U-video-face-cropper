import os
import numpy as np
import cv2
import ffmpeg

from config import VIDEO_PATH, FRAMES_PATH

video_list = os.listdir(VIDEO_PATH)

if not os.path.exists(FRAMES_PATH):
    os.mkdir(FRAMES_PATH)

# get number of frames and duration of video
def video_duration(video):
    probe = ffmpeg.probe(VIDEO_PATH + video)
    time = float(probe['streams'][0]['duration'])
    frame_count = probe['streams'][0]['width']

    return time, frame_count


for index, video in enumerate(video_list):
    print(f"{index}. {VIDEO_PATH}{video}")
    
    time, frame_count = video_duration(video)
    fps = frame_count / time
    random_frames = np.random.randint(1, 199, 10)

    frame_no = 0

    cap = cv2.VideoCapture(VIDEO_PATH + video)

    
    if (cap.isOpened()== False): 
        print("Error opening video  file")
    
    while(cap.isOpened()):
        frame_no += 1

        ret, frame = cap.read()

        if ret:            
            if frame_no in random_frames:
                print('choosen', frame_no)
                video_name = video.split(".")[0]   
                cv2.imwrite(f"{FRAMES_PATH}{video_name}_frame {frame_no}.png", frame)

        else: 
            break
    
    cap.release()   
    cv2.destroyAllWindows()

