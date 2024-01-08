from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse, Response
from torchvision import models
import torch
from torch import nn
import warnings
from PIL import Image
import numpy as np

warnings.filterwarnings("ignore")

app = FastAPI()

app.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def model_initialization(resnet):
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    if resnet:
        model = models.resnet50(weights=None).to(device)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 36) 
        model.load_state_dict(torch.load('saved_models/resnet50_bestacc.pth'))
    else:
        model = models.vgg16(weights=None).to(device)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 36)
        model.load_state_dict(torch.load('saved_models/vgg16_bestacc.pth'))
    model.eval()
    print("model initialized")
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
def prediction(img: np.ndarray) -> Response:
    img = torch.Tensor(img)
    predicted = prediction(img) 

    return JSONResponse(content={"subject": str(predicted)}, status_code=200)



def prediction(dataset):
    dataset = torch.einsum('ijk -> kij', dataset)
    print('sto per predirre veramente')
    output = app.model(dataset)
    print(output)
    _, predicted = torch.max(output, 1)
    print(predicted)
    return predicted