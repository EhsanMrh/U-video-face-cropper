import os
import pandas as pd
import numpy as np
import csv
import cv2
import multiprocessing
from batch_face import RetinaFace
from tqdm import tqdm

from utils import video_duration, check_directory, get_videos, check_rotation, combine_parallel_results
from config import ROOT_DIR, DATASET_PATH, FACES_PATH, LOG_PATH, RESULTS_PATH, ERROR_CODE_NO_VIDEO, NAMES

def init_worker():
    print('Creating networks and loading parameters')
    global face_detector
    face_detector = RetinaFace()

def video_face_cropper(dataset):
    global face_detector

    check_directory(f"{ROOT_DIR}/{LOG_PATH}")

    # Check log files
    processed_videos_path = f"{ROOT_DIR}/{LOG_PATH}/{NAMES['PROCESSED_FILE_NAME']}.csv"
    resume = False
    if os.path.exists(processed_videos_path):
        processed_videos = pd.read_csv(processed_videos_path)
        resume = True

    # Open processed file 

    processed_file_path = ROOT_DIR + f"/{LOG_PATH}/{NAMES['PROCESSED_FILE_NAME']}-{multiprocessing.current_process().name}.csv"
    processed_file = open(processed_file_path, mode='w')
    fieldnames = ["FilePath", "Processed", "Error"]
    processed_file_writer = csv.DictWriter(processed_file, fieldnames=fieldnames)
    processed_file_writer.writeheader()
    processed_file.flush()

    # Open inforamtion file
    information_file_path = ROOT_DIR + f"/{LOG_PATH}/{NAMES['INFORMATION_FILE_NAME']}-{multiprocessing.current_process().name}.csv"
    information_file = open(information_file_path, mode='w')
    fieldnames = ["VideoPath", "FilePath", "Box", "Landmarks", "Confidence"]
    information_file_writer = csv.DictWriter(information_file, fieldnames=fieldnames)
    information_file_writer.writeheader()
    information_file.flush()


    # Loop on videos in dataset
    for video in tqdm(dataset):
        if resume:
            if len(processed_videos[processed_videos['FilePath'] == video]) != 0:
                continue

        error = None
        
        time, frame_count = video_duration(video)
        fps = frame_count / time
        random_frames = np.random.randint(1, 199, 10)
        frame_no = 0

        cap = cv2.VideoCapture(video)
        try:
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        except:
            pass

        if (cap.isOpened() == False): 
            print("Error opening video  file")
            error = ERROR_CODE_NO_VIDEO
            processed_file_writer.writerow({"FilePath": video,
                                            "Processed": False, 
                                            "Error": ERROR_CODE_NO_VIDEO})
            processed_file.flush()
            continue

        rotation_code = check_rotation(video)

        while(cap.isOpened()):
            frame_no += 1

            ret, frame = cap.read()
            if ret: 
                # Get random frame           
                if frame_no in random_frames:
                    
                    # Check Rotation
                    if rotation_code is not None:
                        frame = cv2.rotate(frame, rotation_code)

                    # Face detection
                    face = face_detector(frame, cv=True)
                    if len(face) == 0:
                        # print("!! NO face is detected!")
                        continue
                    if len(face) > 1:
                        # print("!! Two face detected!")
                        continue

                    box, landmarks, confidence = face[0]

                    # Add Margin
                    (x, y, x2, y2) = box
                    margin = int(0.1 * (x2 - x))
                    x = max(x - margin, 0)
                    x2 = min(x2 + margin, frame.shape[1])
                    y = max(y - margin, 0)
                    y2 = min(y2 + margin, frame.shape[0])

                    start_point = (int(x), int(y))
                    end_point = (int(x2), int(y2))

                    # Crop Frame
                    image_cropped = frame[start_point[1]:end_point[1], start_point[0]:end_point[0]]

                    # Save image
                    video_name = video.split('.')[0].split('/')[-1]  
                    cv2.imwrite(f"{ROOT_DIR}/{FACES_PATH}{video_name}_frm{frame_no}.png", image_cropped) 
                    # Write inforamtion
                    information_file_writer.writerow({"VideoPath": video,
                                                        "FilePath": f"{video_name}_frm{frame_no}.png",
                                                        "Box": box, 
                                                        "Landmarks": landmarks,
                                                        "Confidence": confidence})
                    information_file.flush()

            else: 
                break

        if error is None:
            processed_file_writer.writerow({"FilePath": video,
                                            "Processed": True, 
                                            "Error": None})

        cap.release()   
        cv2.destroyAllWindows()

    # Close logs file
    processed_file.close()
    information_file.close()



if __name__ == "__main__":
    num_process = 1

    # Merge logs
    if os.path.exists(f"{ROOT_DIR}/{LOG_PATH}"):
        combine_parallel_results(f"{ROOT_DIR}/{LOG_PATH}", NAMES["PROCESSED_FILE_NAME"])
        combine_parallel_results(f"{ROOT_DIR}/{LOG_PATH}", NAMES["INFORMATION_FILE_NAME"])


    # Check faces directory
    check_directory(f"{ROOT_DIR}/{RESULTS_PATH}")
    check_directory(f"{ROOT_DIR}/{FACES_PATH}")

    list_of_videos = get_videos(DATASET_PATH)
    chunks = np.array_split(np.array(list_of_videos), num_process)
    process_pool = multiprocessing.Pool(processes=num_process, initializer=init_worker)
    pool_output = process_pool.map(video_face_cropper, chunks)

    init_worker()
    video_face_cropper(list_of_videos)

    print("End of Process !")
