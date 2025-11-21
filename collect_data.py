from arm import get_joints, set_state, disengage
import numpy
import cv2
import os
import json
import time

dataset_dir = 'dataset_grip'
os.makedirs(dataset_dir, exist_ok=True)
count = 0


def save_image(frame, count):
    file_name = f'image_{count}.jpg'
    file_path = f'{dataset_dir}/{file_name}'
    cv2.imwrite(file_path, frame)

def save_joints(current_joints, count):
    data = {'joints': current_joints, 'deltas': [0]*6}  # Deltas from ready_to_grab
    with open(f'{dataset_dir}/joints_{count}.json', 'w') as f:
        json.dump(data, f)

set_state('ready_to_grab')
time.sleep(1)

camera = cv2.VideoCapture(0)
cv2.namedWindow("preview")
if camera.isOpened(): 
    rval, frame = camera.read()
else:
    rval = False

while rval:
    disengage()
    cv2.imshow("preview", frame)
    rval, frame = camera.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
    elif key == ord('c'):
        current_joints = get_joints()  # [j1..j6]
        target_joints = current_joints
        save_image(frame, count)
        save_joints(current_joints, count)
        set_state('ready_to_grab')
        count += 1
camera.release()
cv2.destroyWindow("preview")