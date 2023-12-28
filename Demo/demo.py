import streamlit as st
import numpy as np
from PIL import Image
import cv2
from yolo import YOLO

import mediapipe

st.title("Sign Language Recognition Demo")
st.text("Rube Rube Rube")

models_folder = "Demo\yolo-hand-detection-master\models"
print("loading yolo...")
yolo = YOLO(models_folder+"\cross-hands.cfg", models_folder+"\cross-hands.weights", ["hand"])
print("yolo downloaded...")

yolo.size = int(416)
yolo.confidence = 0.2

print("starting webcam...")
import cv2
import streamlit as st

st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
frame_n = 0

while run:
    frame_n+=1
    print(frame_n)
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
    if frame_n%10==0:
        #if img_file_buffer is not None:
        # To read image file buffer with OpenCV:
        #bytes_data = img_file_buffer.getvalue()
        #cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        #print("image_converted")
        
        #st.write(cv2_img.shape)
        width, height, inference_time, results = yolo.inference(frame)
        results.sort(key=lambda x: x[2])
        print(len(results))

        # display hands
        for detection in results:
            id, name, confidence, x, y, w, h = detection
            cx = x + (w / 2)
            cy = y + (h / 2)

            # draw a bounding box rectangle and label on the image
            color = (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "%s (%s)" % (name, round(confidence, 2))
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

        st.image(frame)
        print("picture retrieved")
        print("daje")
else:
    st.write('Stopped')
#img_file_buffer = st.camera_input("Take a picture")

