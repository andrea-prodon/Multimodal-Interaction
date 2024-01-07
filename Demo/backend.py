from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse, Response
import torchvision
import torch
import warnings
from PIL import Image
import numpy as np

warnings.filterwarnings("ignore")

app = FastAPI()

app.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def model_initialization():
    
    model = ... 
    return model


@app.on_event("startup")
def init_model() -> None:
    print('Server startup.')
    app.model = model_initialization()
    print('Start up is over. Server is running')


@app.get('/')
def ping() -> str:
    return "ok"


@app.post('/prediction')
def prediction(file: UploadFile) -> Response:

    # load image
    img = Image.open(file.file)
    img = np.array(img)

    img = preprocessing(img)
    predicted = app.model.predict(img) 

    return JSONResponse(content={"subject": str(predicted)}, status_code=200)



def preprocessing(image: np.ndarray) -> np.ndarray:
    image = ...
    return image

