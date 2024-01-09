import streamlit as st
import numpy as np
from PIL import Image
import cv2
from yolo import YOLO
import mediapipe as mp
#import tensorflow as tf
import torch
from torchvision import datasets, transforms, models
import asyncio
import torch.nn as nn
import httpx
import time


def model_initialization(resnet=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if resnet:
        model = models.resnet50(weights=None).to(device)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 36).to(device)
        model.load_state_dict(torch.load('saved_models/resnet50_bestacc.pth'))
    else:
        model = models.vgg16(weights=None).to(device)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 36).to(device)
        model.load_state_dict(torch.load('saved_models/vgg16_bestacc.pth'))
    model.eval()
    print("model initialized")
    return model


def frame_resize(frame):
    resized_img_array = cv2.resize(frame, (400, 400))
    return resized_img_array

def prediction(model, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = torch.einsum('ijk -> kij', dataset)[None,:].to(device)
    print('sto per predirre veramente')
    output = model(dataset)
    print(output.shape)
    _, predicted = torch.max(output, 1)
    print(predicted)
    
    st.write(predicted)
        
async def main():
    PREDICTION_SERVICE_HOSTNAME = '127.0.0.1'#os.environ["PREDICTION_SERVICE_HOSTNAME"]
    PREDICTION_SERVICE_PORT = '8000'#os.environ["PREDICTION_SERVICE_PORT"]
    st.title("Sign Language Recognition Demo")
    st.text("Rube Rube Rube")

    print("starting webcam...")
    model = model_initialization(False)

    st.title("Webcam Live Feed")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    frame_n = 0


    mphands = mp.solutions.hands
    hands = mphands.Hands()
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    frames = 0

    while run:
        frames += 1
        
        _, frame = cap.read()
        h, w, c = frame.shape

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame)
        hand_landmarks = result.multi_hand_landmarks
        try:
            if hand_landmarks:
                for handLMs in hand_landmarks:
                    x_max = 0
                    y_max = 0
                    x_min = w
                    y_min = h
                    for lm in handLMs.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        if x > x_max:
                            x_max = x
                        if x < x_min:
                            x_min = x
                        if y > y_max:
                            y_max = y
                        if y < y_min:
                            y_min = y
                    
                    meta = y_max//4
                    meta2 = x_max//4

                    # if frames%100 == 0:
                    #     st.image(frame[y_min-20:y_max+20, x_min-20:x_max+20])
                    
                    mano = frame[y_min-meta2:y_max+meta2, x_min-meta:x_max+meta]
                    mano = frame_resize(mano)
                    if frames%10 == 0:
                        st.image(mano)
                        dataset = torch.Tensor(mano)
                        prediction(model,dataset)
                        #task = asyncio.create_task(prediction(model,dataset))
                        #asyncio.run(task)
                        #await task
                        
                    #cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        except:
            pass
        FRAME_WINDOW.image(frame)


if __name__ == '__main__':
    '''
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_initialization()
    img_path = 'asl_dataset/0/hand1_0_bot_seg_1_cropped.jpeg'
    print(device)
    img = torch.einsum('ijk -> kij',torch.Tensor(cv2.imread(img_path)))[None,:].to(device)
    print(img.shape)
    output = model(img)
    print(output)
    '''
    asyncio.run(main())

