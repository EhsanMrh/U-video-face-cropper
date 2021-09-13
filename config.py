import os

SUCCESS = 100
ERROR_VIDEO = 101
ERROR_CODE_NO_VIDEO = 101


ROOT_DIR = os.path.dirname(os.path.abspath("main.py"))
DATASET_PATH = 'videos/'
FACES_PATH = 'faces/'
LOG_PATH = 'logs/'

NAMES = {
    "PROCESSED_FILE_NAME": 'processed_videos_face_cropper',
    "INFORMATION_FILE_NAME": 'information_videos_face_cropper'
}