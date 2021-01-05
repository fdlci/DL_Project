from Main import *
from torchvision import models
import torch.nn as nn

"""Builds the models we are going to train by defining
the layers we are going to train as we use pre-trained 
models."""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def set_parameter_requires_grad(model, feature_extracting):
    """Setting all parameters to require gradient False"""
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def parameters_to_learn(feature_extract, model_ft):
    """Defining the parameters we want to fine-tune (last layer)"""

    print("Paramaters to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

def defining_model_to_train_ResNet(num_classes):
    """Defining the ResNet model, fine-tuning the last two layers"""

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    for param in model_ft.parameters():
        param.requires_grad = True

    for name, child in model_ft.named_children():
        for name2, params in child.named_parameters():
                if 'layer4' not in name+name2 and 'fc' not in name:
                    params.requires_grad = False

    parameters_to_learn(True, model_ft)

    model_ft = model_ft.to(device)
    return model_ft

def defining_model_to_train_VGG16(num_classes):
    """Defining the VGG16 model, fine-tuning the last layer"""

    feature_extract=True
    model_ft = models.vgg16(pretrained=True)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    parameters_to_learn(feature_extract, model_ft)

    model_ft = model_ft.to(device)
   
    return model_ft

def defining_model_to_train_AlexNet(num_classes):
    """Defining the AlexNet model, fine-tuning the last layer"""

    feature_extract = True
    model_ft = models.alexnet(pretrained=True)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    parameters_to_learn(feature_extract, model_ft)

    model_ft = model_ft.to(device)
  
    return model_ft