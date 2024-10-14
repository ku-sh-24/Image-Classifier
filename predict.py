import torchvision
import torch
from torchvision import datasets, transforms,models
from torch import nn,optim
from collections import OrderedDict
import requests
from torch.autograd import Variable

from torch.utils.data import DataLoader

import json
from PIL import Image
import numpy as np
import argparse



# Define a new Classifier
model = models.vgg16(pretrained = True)

for param in model.parameters():
    param.requires_grad = False
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model.input = checkpoint['input_size']
    model.output = checkpoint['output_size']
    model.epochs=checkpoint['epochs']
    model.classifier=checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['model_state'])  
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)
    dim_adjust = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])
                            ])
    pil_image = dim_adjust(pil_image)
    np_image = np.array(pil_image)
    return np_image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    img =process_image(image_path)
    img = torch.from_numpy(img)    
    
    img = img.unsqueeze(0)

    # freeze parameters of the model
    model.eval()
    img=img.to(device)
    with torch.no_grad():
        outputs = model(img)
        ps = torch.exp(outputs)
    top_p, top_class = ps.topk(topk, dim=1)
    idx_to_actual_class = {v:k for k,v in model.class_to_idx.items()}
    top_labels = top_class.tolist()[0]
    top_prop = top_p.tolist()[0]
    flower_classes = [idx_to_actual_class[i] for i in top_labels] 
    return top_prop,flower_classes  


parser = argparse.ArgumentParser(description= 'Predicting for image Classifier')
parser.add_argument('input_img',default='./flowers/test/13/image_05775.jpg',action='store')
parser.add_argument('--checkpoint',default='./checkpoint.pth',action='store')
parser.add_argument('--top_k',type=int,default=5,action='store')
parser.add_argument('--cat_to_name',default='cat_to_name.json',action='store')
parser.add_argument('--gpu',default='gpu',action='store')

input_argu = parser.parse_args()
input_img = input_argu.input_img
checkpoint = input_argu.checkpoint
topk =input_argu.top_k
cat_to_name = input_argu.cat_to_name
device = input_argu.gpu

def main():
    model=load_checkpoint(checkpoint)
    device = input_argu.gpu
    model.to(device)
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    top_probs,top_classes = predict('./flowers/test/13/image_05775.jpg',model)
    flower_names = [cat_to_name.get(i) for i in top_classes]   
    predicts = {name: prob for name, prob in zip(flower_names, top_probs)}
    print(predicts)

if __name__ == "__main__":
    main()