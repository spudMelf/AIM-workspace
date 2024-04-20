import io
import os
import torchvision.transforms as transforms
from torchvision import models
import torch
import torch.nn as nn
from PIL import Image

# preprocess our image
def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),           
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return my_transforms(image).unsqueeze(0)

# prediction
class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet, self).__init__()
        self.resnet = models.resnet18(weights=None)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

PATH = 'mel_spectrogram_model_0.pt'
device = torch.device('cpu')
model = CustomResNet(num_classes=2)
model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = model.forward(tensor)
    _, predicted = torch.max(outputs.data, 1)
    if predicted == 1:
        return [1, 'spoof']
    elif predicted == 0:
        return [0, 'bona-fide']
    else:
        return [-1, 'ERROR: prediction error in get_prediction()']
    

    
