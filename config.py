import os

SUCCESS = 100
ERROR_VIDEO = 101
ERROR_CODE_NO_VIDEO = 101


ROOT_DIR = os.path.dirname(os.path.abspath("main.py"))
DATASET_PATH = 'videos'
DATASET_NAME = DATASET_PATH.split('/')[-1]
RESULTS_PATH = 'results/'
FACES_PATH = f'{RESULTS_PATH}faces_{DATASET_NAME}/'
LOG_PATH = f'{RESULTS_PATH}logs_{DATASET_NAME}/'

NAMES = {
    "PROCESSED_FILE_NAME": 'processed_videos_face_cropper',
    "INFORMATION_FILE_NAME": 'information_videos_face_cropper'
}