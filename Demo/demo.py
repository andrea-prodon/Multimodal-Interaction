import streamlit as st
from streamlit_image_select import image_select
import numpy as np
from PIL import Image

st.title("Sign Language Recognition Demo")
st.text("Rube Rube Rube")

img = image_select(
    label="Select a cat",
    images=[
        "images/demarsico.jpg",
        "https://bagongkia.github.io/react-image-picker/0759b6e526e3c6d72569894e58329d89.jpg",
        Image.open("images/a1.jpg"),
        np.array(Image.open("images/a1.jpg")),
    ],
    captions=["A cat", "Another cat", "Oh look, a cat!", "Guess what, a cat..."],
)