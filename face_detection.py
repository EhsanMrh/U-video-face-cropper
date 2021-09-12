import os
import cv2
from batch_face import RetinaFace

from config import FRAMES_PATH, FACES_PATH, FACE_DETECTED_PATH

# Create directories
if not os.path.exists(FACES_PATH):
    os.mkdir(FACES_PATH)

if not os.path.exists(FACE_DETECTED_PATH):
    os.mkdir(FACE_DETECTED_PATH)

frame_list = os.listdir(FRAMES_PATH)
frame_list.sort()

detector = RetinaFace()

for index, frame_name in  enumerate(frame_list):
    print(f"{index}: {frame_name}")
    # Load image
    orginal = cv2.imread(FRAMES_PATH + frame_name)
    img = orginal.copy()

    # Face detection
    face = detector(img, cv=True)
    if len(face) == 0:
        print("NO face is detected!")
        continue
    box, landmarks, score = face[0]

    # Add Margin
    (x, y, x2, y2) = face[0][0]
    margin = int(0.1 * (x2 - x))
    x = max(x - margin, 0)
    x2 = min(x2 + margin, img.shape[1])
    y = max(y - margin, 0)
    y2 = min(y2 + margin, img.shape[0])


    # Draw box around face
    # start_point = (int(box[0]), int(box[1]))
    start_point = (int(x), int(y))
    print("start point", start_point)
    # end_point = (int(box[2]), int(box[3]))
    end_point = (int(x2), int(y2))
    print("end_point", end_point)
    color = (255, 0, 0)
    thickness = 2
    
    cv2.rectangle(img, start_point, end_point, color, thickness)
    # box_image = cv2.circle(img, (int(landmarks[0][0]), int(landmarks[0][1])), color = (0, 0, 255) ,radius=5, thickness = -1)

    # Save image
    image_cropped = orginal[start_point[1]:end_point[1], start_point[0]:end_point[0]]
    cv2.imwrite(FACES_PATH + 'cropped-' + frame_name, image_cropped) 

    cv2.imwrite(FACE_DETECTED_PATH + frame_name, img) 




    