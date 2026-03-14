
import os
from PIL import Image
import torchvision.transforms as T


def pil_load(path):
return Image.open(path).convert('RGB')


def preprocess_pil(img, size=224):
transform = T.Compose([
T.Resize((size,size)),
T.ToTensor(),
T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
return transform(img)
