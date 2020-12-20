import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import os

num_classes = 12

def loading_saved_model():
    # model_ft = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)
    # model_ft.eval()
    # num_ftrs = model_ft.classifier[6].in_features
    # model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    # model_ft.load_state_dict(torch.load('Models/VGG16_50_Oxford.pt', map_location='cpu'))
    # model_ft.eval()

    model_ft = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    model_ft.eval()

    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    model_ft.load_state_dict(torch.load('AlexNet.pt', map_location='cpu'))
    model_ft.eval()   

    # model_ft = torch.hub.load('pytorch/vision:v0.6.0', 'resnet101', pretrained=True)
    # model_ft.eval()
    # num_ftrs = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_ftrs, 11)

    # model_ft.load_state_dict(torch.load('ResNet101.pt', map_location='cpu'))
    # model_ft.eval() 

    return model_ft

def getting_the_test_data():

    data_transforms = {'Test Set': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # data_dir = 'Paris'
    data_dir = 'Oxford'

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['Test Set']}
    dataloaders_test = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=11,
                                                shuffle=True, num_workers=4)
                for x in ['Test Set']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['Test Set']}
    class_names = image_datasets['Test Set'].classes

    Total_size = dataset_sizes['Test Set']
    return dataloaders_test, class_names, Total_size

def getting_predictions(dataloaders_test, model_ft):
    inputs_list = []
    labels_list = []
    for inputs, labels in dataloaders_test['Test Set']:
        labels_list += labels.tolist()
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)
        inputs_list += preds.tolist()

    return np.array(inputs_list), np.array(labels_list)

def computing_accuracy(inputs_list, labels_list, class_names, Total_size):
    accuracies = {}
    acc_total = 0
    for i in range(num_classes):
        labels_index = np.where(labels_list == i)[0]
        acc = 0
        for j in labels_index:
            if inputs_list[j] == labels_list[j]:
                acc += 1
        acc_total += acc
        accuracies[class_names[i]] = acc/len(labels_index)
    accuracies['Total'] = acc_total/Total_size
    return accuracies

def main():

    # Loading the model
    model_ft = loading_saved_model()

    # Getting the data
    dataloaders_test, class_names, Total_size = getting_the_test_data()

    # Pridicting landmarks
    inputs_list, labels_list = getting_predictions(dataloaders_test, model_ft)

    # Computing the accuracy
    accuracy = computing_accuracy(inputs_list, labels_list, class_names, Total_size)

    return accuracy

accuracy = main()
print(accuracy)