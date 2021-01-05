import torch
from torchvision import datasets, transforms
import numpy as np
import os
from Defining_models import defining_model_to_train_ResNet, defining_model_to_train_VGG16, defining_model_to_train_AlexNet
from Main import number_of_classes

"""Makes predictions on the test set loading the saved models"""  

def loading_saved_model(num_classes, model_chosen, model_name):
    """Loads the saved model"""

    # Loading VGG16
    if model_chosen == 'VGG16':
        model = defining_model_to_train_VGG16(num_classes)
        model.load_state_dict(torch.load(model_name, map_location = 'cpu'))
        model.eval()

    # Loading AlexNet
    elif model_chosen == 'AlexNet':
        model = defining_model_to_train_AlexNet(num_classes)
        model.load_state_dict(torch.load(model_name, map_location = 'cpu'))
        model.eval()

    # Loading ResNet
    elif model_chosen == 'ResNet':
        model = defining_model_to_train_ResNet(num_classes)
        model.load_state_dict(torch.load(model_name, map_location = 'cpu'))
        model.eval()

    return model

def getting_the_test_data(data_dir):
    """Gets the test data"""

    data_transforms = {'Test Set': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

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
    """Predicts the class of every image of the test set"""

    inputs_list = []
    labels_list = []

    for inputs, labels in dataloaders_test['Test Set']:
        labels_list += labels.tolist()
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)
        inputs_list += preds.tolist()

    return np.array(inputs_list), np.array(labels_list)

def computing_accuracy(inputs_list, labels_list, class_names, Total_size, num_classes):
    """Computes the accuracy for every landmark and the overall accuracy"""

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

def main(data_dir, model_chosen, model_name):

    # Number of classes
    num_classes = number_of_classes(data_dir)

    # Loading the model
    model_ft = loading_saved_model(num_classes, model_chosen, model_name)

    # Getting the data
    dataloaders_test, class_names, Total_size = getting_the_test_data(data_dir)

    # Pridicting landmarks
    inputs_list, labels_list = getting_predictions(dataloaders_test, model_ft)

    # Computing the accuracy
    accuracy = computing_accuracy(inputs_list, labels_list, class_names, Total_size, num_classes)

    return accuracy


# data_dir = 'Paris'
data_dir = 'Oxford'

model_chosen = 'VGG16'
model_name = 'Models/VGG16_75_Oxford.pt'

# model_chosen = 'ResNet'
# model_name = 'Models/ResNet50_Paris.pt'

# model_chosen = 'AlexNet'
# model_name = 'Models/AlexNet_Paris.pt'

accuracy = main(data_dir, model_chosen, model_name)
print(accuracy)