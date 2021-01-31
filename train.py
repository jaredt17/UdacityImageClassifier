# Udacity Command Line App - Image Classifier
# Jared Teller
# 1/29/2021

# IMPORTS
# from workspace_utils import keep_awake # use this if running a long training
import time # time just in case
import sys
import torch # need torch, torchvision, models, datasets, transforms, nn, optim
from torchvision import models, datasets, transforms
from torch import nn, optim
import matplotlib.pyplot as plt # need matplotlib stuff

from collections import OrderedDict # need collections OrderedDict
import numpy as np # need numpy as np
from PIL import Image # PIL Images

# import our custom argparser
from get_input_args import get_input_args_train

# for checking directories
from os.path import isdir

# MAIN FUNCTION
def main():

    # Set up model options
    # resnet18 = models.resnet18(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)
    vgg11 = models.vgg11(pretrained=True)
    vgg13 = models.vgg13(pretrained=True)


    # get the input arguments
    in_arg = get_input_args_train() # options are .dir, .arch, .save_dir, .learning_rate, .hidden_units, .epochs, .gpu
    
    # set up model architecture
    model_architecture = in_arg.arch

    data_dir = in_arg.dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets\
    # for training transforms we will want randomization and horizontal flips
    training_transforms = transforms.Compose([transforms.RandomRotation(30), 
                                              transforms.RandomResizedCrop(224), 
                                              transforms.RandomHorizontalFlip(), 
                                              transforms.ToTensor(), 
                                              transforms.Normalize([0.485, 0.456, 0.406], 
                                                                   [0.229, 0.224, 0.225])])
    
    # for valid dataset our transforms will not be randomized, just centercroped and resized, then normalized and set to tensors
    validation_transforms = transforms.Compose([transforms.Resize(256), 
                                                transforms.CenterCrop(224), 
                                                transforms.ToTensor(), 
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # for test transforms, same as above
    test_transforms = transforms.Compose([transforms.Resize(256), 
                                          transforms.CenterCrop(224), 
                                          transforms.ToTensor(), 
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    # image_datasets = need 3 different ones, not sure why this is here
    train_data = datasets.ImageFolder(train_dir, transform = training_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = validation_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    # dataloaders =
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = 64)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = 64)
    
    # Set the model
    if model_architecture == "vgg11":
        model = vgg11
    elif model_architecture == "vgg13":
        model = vgg13
    elif model_architecture == "vgg16":
        model=vgg16
    else:
        print("Please enter a valid model architecture. Options are vgg11, vgg13, and vgg16.")
        sys.exit("Invalid Model.")
    # model = eval("models.{}(pretrained=True)".format(model_architecture))

    model.name = model_architecture

    # print(model)
    
    # Create our classifier
    # freeze parameteres so we dont backpropagate
    for param in model.parameters():
        param.requires_grad = False

    # get input features from model
    input_features = model.classifier[0].in_features
    
    # set hidden_units from the arguments, default is 4096
    hidden_units = in_arg.hidden_units
    
    # defining our classifier as a sequential ordered dict, as seen in lessons
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_features, hidden_units, bias = True)), 
        ('relu', nn.ReLU()), 
        ('dropout', nn.Dropout(p=0.5)), 
        ('fc2', nn.Linear(hidden_units, 102, bias = True)), 
        ('output', nn.LogSoftmax(dim = 1))]))

    model.classifier = classifier

    # print(model)
    
    # initialize device as CPU
    device = torch.device("cpu")
    # set up cuda or cpu since we will need some gpu time
    if in_arg.gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
            print("GPU selected, but there is no GPU available.")
    print("Model running on: ", device)
    
    model.to(device)
    
    # we need loss criterion and to choose an optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.learning_rate) # going to start out with adam

    print("Model Architecture: ", model.name)
    print("Input Features:", input_features)
    print("Hidden units: ", in_arg.hidden_units)
    print("Learning Rate: ", in_arg.learning_rate)
    print("Epochs: ", in_arg.epochs)

    # Begin training the model
    train_nn(model, device, in_arg.epochs, train_loader, valid_loader, test_loader, optimizer, criterion)
    

    learn_rate = in_arg.learning_rate

    # create_checkpoint(model, in_features, hidden_units, train_data, optimizer):
    # Save the model checkpoint
    if in_arg.save_dir != None:
        save_directory = in_arg.save_dir
        if isdir(save_directory):
            create_checkpoint(model, input_features, in_arg.hidden_units, train_data, optimizer, learn_rate)
            print("Checkpoint Created at path: ", save_directory)
    else:
        print("Model will not be saved since no directory was specified using --save_dir (directory_here)")

# END MAIN FUNCTION

# validate function will take in the model, a specific dataset, and the loss criterion and return the loss and accuracy on that data
def validate(model, device, data_loader, criterion):
    test_loss = 0
    accuracy = 0
    for ii, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        #start the timer
        start = time.time()
        
        # start the feedforward process
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        
        test_loss += loss.item()
        ps_data = torch.exp(outputs)
        equality = (labels.data == ps_data.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean() # need a float tensor
        
    return test_loss, accuracy # send back results

# Train the neural network with the given number of epochs
def train_nn(model, device, epochs, train_loader, valid_loader, test_loader, optimizer, criterion, ):
    # now we need to set our epochs, our steps, and how often we want to print some output
    print_step = 25 # print output after 25 images
    steps = 0

    print("TRAINING STARTED...\n")

    for epoch in range(epochs):
        run_loss = 0
        # set training mode for pytorch
        model.train()

        for ii, (inputs, labels) in enumerate(train_loader): # running on our loaded training dataset

            # reset the gradients
            optimizer.zero_grad()

            steps += 1

            #move the inputs and labels to gpu/cpu
            inputs, labels = inputs.to(device), labels.to(device)

            # forward pass
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.item()

            if steps % print_step == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss, accuracy = validate(model, device, valid_loader, criterion)

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {run_loss/print_step:.3f}.. "
                      f"Test loss: {valid_loss/len(test_loader):.3f}.. "
                      f"Test accuracy: {accuracy/len(test_loader):.3f}")
                run_loss = 0
                model.train()
    print("\nTRAINING COMPLETE.")
# END TRAINING FUNCTION

# Testing function, only used if code is modified, not a requirement
def test_nn(model, device, test_loader, criterion):
    # we should be able to just use our validation function we already created and pass in the test_loader data...
    with torch.no_grad():
        model.eval()
        test_loss, accuracy = validate(model, device, test_loader, criterion)
    print(f"Accuracy achieved on test images is: {(accuracy/len(test_loader)) * 100 :.2f}%")
# END TEST NN Function

# Creates a checkpoint of the model if there is a save directory specified
def create_checkpoint(model, in_features, hidden_units, train_data, optimizer, learn_rate):

    # Save the class indexes 
    model.class_to_idx = train_data.class_to_idx # save our training dataset

    # create the checkpoint structure
    checkpoint = {
        'model_name': model.name,
        'input_size': in_features,
        'output_size': 102,
        'hidden_units': hidden_units,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx,
        'classifier': model.classifier,
        'learning_rate': learn_rate
    }
    torch.save(checkpoint, 'checkpoint.pth')
# END CREATE CHECKPOINT FUNCTION

if __name__ == "__main__":
    main()