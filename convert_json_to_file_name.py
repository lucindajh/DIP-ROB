import glob
import cv2
import json
import os

dir_path = 'dataset_grip_2'
os.makedirs(dir_path, exist_ok=True)

images = glob.glob("dataset_grip/*.jpg")
count = 0
for image in images:
    img = cv2.imread(image)
    data = {
        'joints' : [0,0,0,0,0],
        'deltas' : [0,0,0,0,0]
    }
    with open(f'dataset_grip/joints_{count}.json', 'r') as file:
        data = json.load(file)
    cv2.imwrite(f'{dir_path}/{data['joints']}.jpg', img)
    count += 1




