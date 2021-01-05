from Main import *
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from Defining_models import defining_model_to_train_VGG16

"""Builds and trains the model"""

# Defining the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('The model is training on: ' + str(device))

def train_pre_trained_model_VGG16(pre_trained_model):
    """For a built model, trains the model"""

    # Defining the loss
    criterion = nn.CrossEntropyLoss()

    # Optimizing all parameters
    optimizer_ft = optim.SGD(pre_trained_model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Loading the Data
    dataloaders = defining_data()[0]

    # Training the model
    model_ft = train_model(pre_trained_model, dataloaders, criterion, optimizer_ft, exp_lr_scheduler)

    return model_ft


def main_train_VGG16(num_classes):
    """Builds and trains the model"""

    # Defining the model
    model_ft = defining_model_to_train_VGG16(num_classes)

    # Training the model
    model_trained = train_pre_trained_model_VGG16(model_ft)

    return model_trained

#Paris: num_classes = 11
#Oxford: num_classes = 12

num_classes = 12
model_ft = main_train_VGG16(num_classes)

# Saving the model
torch.save(model_ft.state_dict(),"VGG16_Oxford_25.pt")