import os
import pandas as pd
from tqdm import tqdm
import ffmpeg
import cv2


from config import ROOT_DIR


# Check and Create Directory
def check_directory(path):
    if not os.path.exists(path):
        os.mkdir(path)


# get number of frames and duration of video
def video_duration(video):
    probe = ffmpeg.probe(video)
    time = float(probe['streams'][0]['duration'])
    frame_count = probe['streams'][0]['width']

    return time, frame_count

# To get videos from dataset directory
def get_videos(dataset):
    list_of_videos = []

    print(f"Dataset: {dataset}")
    for root, dirs, files in os.walk(os.path.join(ROOT_DIR, dataset)):
        list_of_videos += [os.path.join(root, file) for file in files if file.find(".mp4") != -1]
    
    return list_of_videos

def check_rotation(video_path):
    """
    :param video_path: path of input video
    :return: returns the rotation angle of the input video
    """
    # this returns meta-data of the video file in form of a dictionary
    rotate_code = None
    meta_dict = ffmpeg.probe(video_path)
    keys = meta_dict['streams'][0]['tags'].keys()

    if 'rotate' in keys:
        # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
        # we are looking for
        if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
            rotate_code = cv2.ROTATE_90_CLOCKWISE
        elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
            rotate_code = cv2.ROTATE_180
        elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
            rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE

    return rotate_code

def combine_parallel_results(read_path, csv_prefix):
    csv_files_read = os.listdir(read_path)
    csv_files = []
    for csv_file in csv_files_read:
        csv_path = os.path.join(read_path, csv_file)
        if csv_prefix in csv_path:
            try:
                csv_files.append(pd.read_csv(csv_path))
            except pd.errors.EmptyDataError:
                print('[INFO] Empty CSV file Error for file {}'.format(os.path.join(csv_path, csv_file)))

    data_df = csv_files[0]
    for df in tqdm(csv_files[1:]):
        data_df = data_df.append(df)
    data_df.to_csv(read_path + '/' + csv_prefix + '.csv', index=False)

    print(f"+ {csv_prefix}.csv saved!")
