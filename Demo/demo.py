import streamlit as st
import numpy as np
from PIL import Image
import cv2

import mediapipe

st.title("Sign Language Recognition Demo")
st.text("Rube Rube Rube")

picture = st.camera_input("Take a picture of your hands inside the rectangle")

if picture:
    st.image(picture)

