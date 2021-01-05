from Main import number_of_classes, defining_data, train_model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from Defining_models import defining_model_to_train_VGG16

"""Builds and trains the model"""

# Defining the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('The model is training on: ' + str(device))

def train_pre_trained_model_VGG16(pre_trained_model, data_dir):
    """For a built model, trains the model"""

    # Defining the loss
    criterion = nn.CrossEntropyLoss()

    # Optimizing all parameters
    optimizer_ft = optim.SGD(pre_trained_model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Loading the Data
    dataloaders = defining_data(data_dir)[0]

    # Training the model
    model_ft = train_model(pre_trained_model, data_dir, dataloaders, criterion, optimizer_ft, exp_lr_scheduler)

    return model_ft


def main_train_VGG16(data_dir):
    """Builds and trains the model"""

    # Number of Classes
    num_classes = number_of_classes(data_dir)

    # Defining the model
    model_ft = defining_model_to_train_VGG16(num_classes)

    # Training the model
    model_trained = train_pre_trained_model_VGG16(model_ft, data_dir)

    return model_trained

# data_dir = 'Paris'
data_dir = 'Oxford'

model_ft = main_train_VGG16(data_dir)

# Saving the model
torch.save(model_ft.state_dict(),"VGG16_Oxford_25.pt")