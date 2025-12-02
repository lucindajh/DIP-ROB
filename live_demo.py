import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import PIL.Image
import numpy as np
import xarm
import arm
import glob
import json 
import time


model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)

model.load_state_dict(torch.load('best_model_resnet18.pth', map_location=torch.device('cpu')))
model = model.eval()

mean = torch.Tensor([0.485, 0.456, 0.406])
std = torch.Tensor([0.229, 0.224, 0.225])

normalize = torchvision.transforms.Normalize(mean, std)

def preprocess(image):
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

import cv2
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if camera.isOpened():
    rval, frame = camera.read()
else:
    rval = False
while rval:
    arm.set_state('ready_to_grab')

    cv2.imshow("preview", frame)

    
    input_image = preprocess(frame)
    
    
    with torch.no_grad():
      output = model(input_image)
    
    probabilities = F.softmax(output, dim=1)
    
    print(probabilities.flatten()[0])

    prob_grip = float(probabilities.flatten()[0])

    if prob_grip < 0.6:
        arm.adjust_grip_state(1)
    else:
        arm.set_state('ready_to_move')
        break
    #time.sleep(0.001)
    rval, frame = camera.read()
    
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

time.sleep(1)
arm.disengage()
