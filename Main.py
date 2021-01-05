from __future__ import print_function, division
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
plt.ion()

"""Loads the training and validation sets.
Trains the network"""

def number_of_classes(data_dir):
    """Returns the number of classes"""
    if data_dir == 'Paris':
        num_classes = 11
    elif data_dir == 'Oxford':
        num_classes = 12
    return num_classes

def defining_data(data_dir):
    """Loads the training set and validation set"""

    data_transforms = {
        'Training Set': transforms.Compose([
            transforms.RandomResizedCrop(224), # Crop the given image to random size and aspect ratio
            transforms.RandomHorizontalFlip(), # Horizontally flip the given image randomly with a given probability
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Validation Set': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['Training Set', 'Validation Set']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['Training Set', 'Validation Set']}

    return dataloaders, image_datasets

def train_model(model, data_dir, dataloaders, criterion, optimizer, scheduler, num_epochs=50):
    """Proceeds with model training"""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_datasets = defining_data(data_dir)[1]

    dataset_sizes = {x: len(image_datasets[x]) for x in ['Training Set', 'Validation Set']}

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    training_loss = []
    validation_loss = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['Training Set', 'Validation Set']:
            if phase == 'Training Set':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]: #depends on the number of batches chosen
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'Training Set'): #sets gradient calculations to on or off
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # training_loss.append(loss)

                    # backward + optimize only if in training phase
                    if phase == 'Training Set':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'Training Set':
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'Training Set':
                training_loss.append(epoch_loss)
            else:
                validation_loss.append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'Validation Set' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    # Plotting the validation loss and training loss
    print('validation loss: ' + str(validation_loss))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # plot the training and validation loss
    plt.figure()
    plt.plot(training_loss, 'b', label='Training Loss')
    plt.plot(validation_loss, 'r', label='Validation Loss')
    plt.title('ResNet50: Variations of the training and validation loss') #Change Title for every model
    plt.legend()
    plt.show()
    plt.savefig('ResNet50_Paris.png') #Change title for every model

    return model