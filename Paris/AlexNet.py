from Main import *
import torchvision
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
model.eval()

def defining_model_to_train():

    feature_extract = True
    num_classes = 11

    model_ft = models.alexnet(pretrained=True)
    num_ftrs = model_ft.classifier[6].in_features
    set_parameter_requires_grad(model_ft, feature_extract)
    model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)


    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    params_to_update = model_ft.parameters()
    print("Params to learn:")
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

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    dataloaders = defining_data()[0]

    model_ft = train_model(model_ft, dataloaders, criterion, optimizer_ft, exp_lr_scheduler)
    
    return model_ft

model_ft = defining_model_to_train()

torch.save(model_ft.state_dict(),"AlexNet.pt")