import streamlit as st
import numpy as np
from PIL import Image
import cv2
from yolo import YOLO
import mediapipe as mp
import tensorflow as tf
import torch
from torchvision import datasets, transforms, models

def model_initialization():
    print("entrato")
    model = models.vgg16(pretrained=True).to(device)
    model = torch.load('saved_models/vgg16_bestacc.pth')
    print("model initialized")
    return model

def frame_resize(frame):
    resized_img_array = cv2.resize(frame, (400, 400))
    return resized_img_array

def prediction(dataset):
    model = model_initialization()
    for data in dataset:
        predictions = model(data)
        st.write(predictions)
        print(predictions)

st.title("Sign Language Recognition Demo")
st.text("Rube Rube Rube")

models_folder = "Demo\yolo-hand-detection-master\models"
print("loading yolo...")
yolo = YOLO(models_folder+"\cross-hands.cfg", models_folder+"\cross-hands.weights", ["hand"])
print("yolo downloaded...")

yolo.size = int(416)
yolo.confidence = 0.2

print("starting webcam...")

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
                if frames%100 == 0:
                    st.image(mano)
                    dataset = tf.data.Dataset.from_tensor_slices(mano).batch(1)
                    prediction(dataset)
                #cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    except:
        pass
    FRAME_WINDOW.image(frame)

