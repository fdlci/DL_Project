from Main import *
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from Defining_models import defining_model_to_train_AlexNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def train_pre_trained_model_AlexNet(pre_trained_model):

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(pre_trained_model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    dataloaders = defining_data()[0]

    model_ft = train_model(pre_trained_model, dataloaders, criterion, optimizer_ft, exp_lr_scheduler)

    return model_ft

def main_train_AlexNet(num_classes):

    #Defining the model
    model_ft = defining_model_to_train_AlexNet(num_classes)

    #Training the model
    model_trained = train_pre_trained_model_AlexNet(model_ft)

    return model_trained

#Paris: num_classes = 11
#Oxford: num_classes = 12

num_classes = 12
model_ft = main_train_AlexNet(num_classes)

torch.save(model_ft.state_dict(),"AlexNet.pt")