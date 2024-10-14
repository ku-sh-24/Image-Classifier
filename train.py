# Imports here
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

def data_initialize(data):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485,0.456,0.406],
                                                            [0.229,0.224,0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485,0.456,0.406],
                                                                    [0.229,0.224,0.225])])

    testing_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485,0.456,0.406],
                                                                    [0.229,0.224,0.225])])

    # TODO: Load the datasets with ImageFolder
    training_datasets = datasets.ImageFolder(train_dir,transform=training_transforms)
    validation_datasets = datasets.ImageFolder(valid_dir,transform=validation_transforms)
    testing_datasets = datasets.ImageFolder(test_dir,transform=testing_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    training_dataloaders = DataLoader(training_datasets,batch_size=64,shuffle=True)
    validation_dataloaders = DataLoader(validation_datasets,batch_size=64,shuffle=True)
    testing_dataloaders = DataLoader(testing_datasets,batch_size=64,shuffle=True)
    
    return training_dataloaders,validation_dataloaders,testing_dataloaders,training_datasets

def initialize_model(arch,hidden_units,lr,dropout,device):
    device = 'cuda'
    if arch == 'vgg16':
        model = models.vgg16(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False

    # Define a new Classifier
    model.classifier=nn.Sequential(OrderedDict([
                                        ('input_layer',nn.Linear(25088,hidden_units)),
                                        ('relu',nn.ReLU()),
                                        ('dropout',nn.Dropout(0.3)),
                                        ('hidden_layer1',nn.Linear(hidden_units,128)),
                                        ('relu1',nn.ReLU()),
                                        ('output_layer',nn.Linear(128,102)),
                                        ('output',nn.LogSoftmax(dim=1))
                                    ]))
        
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
    model.to(device)

    return model,criterion,optimizer


parser = argparse.ArgumentParser(description= 'Training for image Classifier')
parser.add_argument('data_dir',default='flowers',action='store')
parser.add_argument('--save_dir',default='./checkpoint.pth',action='store')
parser.add_argument('--arch',default='vgg16',action='store')
parser.add_argument('--epochs',type=int,default=3,action='store')
parser.add_argument('--print_every',type=int,default=20,action='store')
parser.add_argument('--lr',type=float,default=1e-3,action='store')
parser.add_argument('--dropout',type=float,default=0.3,action='store')
parser.add_argument('--hidden_layers',type=int,default=256,action='store')
parser.add_argument('--gpu',default='gpu',action='store')

input_argu = parser.parse_args()
data_path = input_argu.data_dir
checkpoint = input_argu.save_dir
arch =input_argu.arch
epochs = input_argu.epochs
print_every = input_argu.print_every
lr = input_argu.lr
dropout = input_argu.dropout
hidden_layers = input_argu.hidden_layers
device = input_argu.gpu

def main():
    training_dataloaders,validation_dataloaders,testing_dataloaders,training_datasets = data_initialize(data_path)
    model,criterion,optimizer = initialize_model(arch,hidden_layers,lr,dropout,device)
    # move the model parameters to the 'gpu' memory

    for epoch in range(epochs):
        training_loss = 0
        step = 0
        for images,labels in training_dataloaders:
            step += 1
            # move the images and labels to the GPU
            images,labels= images.to(device),labels.to(device)

            log_values = model.forward(images)
            loss = criterion(log_values,labels)

            # clear gradient values
            optimizer.zero_grad()
            # find the new weight values
            loss.backward()
            # update the weights
            optimizer.step()

            training_loss += loss.item()

            if step%print_every == 0:
                valid_loss = 0
                valid_accuracy = 0
                # turning off dropout
                model.eval()
                # turn off gradients
                with torch.no_grad():
                    for images,labels in validation_dataloaders:
                        images,labels= images.to(device),labels.to(device)
                        log_vals = model.forward(images)
                        val_loss = criterion(log_vals,labels)
                        valid_loss += val_loss.item()

                        # calculating accuracy
                        ps = torch.exp(log_vals)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy+= torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {training_loss/print_every:.3f}.. "
                    f"Valid loss: {valid_loss/len(validation_dataloaders):.3f}.. "
                    f"Valid accuracy: {(valid_accuracy/len(validation_dataloaders))*100:.3f}%")
                training_loss=0
                model.train()
    # TODO: Save the checkpoint 
    model.class_to_idx = training_datasets.class_to_idx
    checkpoint = {'input_size' : 25088,
                'output_size':102,
                'epochs':epochs,
                'optimizer_state':optimizer.state_dict(),
                'classifier':model.classifier,
                'model_state':model.state_dict(),
                'class_to_idx':training_datasets.class_to_idx}
    torch.save(checkpoint, 'checkpoint.pth')
    print('Training and Saving model success!')

if __name__ == "__main__":
    main()