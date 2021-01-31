# Udacity Command Line App - Image Classifier
# Jared Teller
# 1/29/2021

# IMPORTS
import time
# need collections OrderedDict
from collections import OrderedDict
# need numpy as np
import numpy as np

import torch # need torch, torchvision, models, datasets, transforms, nn, optim
from torchvision import models, datasets, transforms
from torch import nn, optim

# PIL Images
from PIL import Image
# need matplotlib stuff
import matplotlib.pyplot as plt
import sys
import json # for loading cat_to_name
# import our custom argparser
from get_input_args import get_input_args_predict
# for checking directories
from os.path import isdir

# with open('cat_to_name.json', 'r') as f:
#    cat_to_name = json.load(f)

# load our torch checkpoint to set up the model and optimzer
def load_checkpoint(filepath):
    # preload models
    vgg16 = models.vgg16(pretrained=True)
    vgg11 = models.vgg11(pretrained=True)
    vgg13 = models.vgg13(pretrained=True)
    
    checkpoint = torch.load(filepath)
    if checkpoint['model_name'] == "vgg11":
        model = vgg11
    elif checkpoint['model_name'] == "vgg13":
        model = vgg13
    elif checkpoint['model_name'] == "vgg16":
        model = vgg16    

    # freeze params again
    for param in model.parameters():
        param.requires_grad = False
    
    # checkpoint information loading
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.input_size = checkpoint['input_size']
    model.output_size = checkpoint['output_size']

    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    # we can use the same transforms as earlier
    image_transforms = transforms.Compose([transforms.Resize(256), 
                                            transforms.CenterCrop(224), 
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    loaded_image = image_transforms(Image.open(image)) # import Image from PIL and apply transform
    
    # For some reason this all caused errors, and just applying the transform allows imshow to work properly...
    # convert to numpy array
    # np_image = np.array(loaded_image)/255
    
    # apply transpose to set color channel first
    # np_image = np_image.transpose(2,0,1)
    
    return loaded_image

# ------------------ Prediction Function ------------------------
def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.to(device) # move model to appropriate device
    
    model.eval()
    
    # load the numpy image array/tensor into a torch variable so we can run it through our nn
    pytorch_img = torch.from_numpy(np.expand_dims(process_image(image_path), axis=0)).type(torch.FloatTensor).to(device)
    
    # run through the model
    log_probability = model.forward(pytorch_img)
    ps = torch.exp(log_probability).data # need to apply exponent since we were in log form
    
    # use top K algorithm to return the highest 5 probabilities
    probs_top5 = ps.topk(topk)[0]
    index_top = ps.topk(topk)[1]
    
    cpu_top5 = probs_top5.cpu().numpy()
    cpu_index_np = index_top.cpu().numpy()

    # convert to np arrays instead of tensors
    cpu_top5_res = np.array(cpu_top5)[0]
    cpu_index_top = np.array(cpu_index_np[0])
    
    # invert (swap key and value pairs)
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    # build classes list based on the idx to class we saved
    classes_list = []
    for index in cpu_index_top:
        classes_list += [idx_to_class[index]]
    
    return cpu_top5_res, classes_list
# END PREDICT FUNCTION

def category_mapper(probs, classes, cat_to_name_path):
    with open(cat_to_name_path, 'r') as f:
        cat_to_name = json.load(f)

    flowers = []
    for i in classes:
        flowers += [cat_to_name[i]]

    for x in range(len(flowers)):
        print("Class: ", flowers[x], "...  Probability: ", probs[x] * 100, "%")

    return flowers


# --------------- MAIN FUNCTION --------------
def main():

    # data_dir = 'flowers'
    # train_dir = data_dir + '/train'
    # valid_dir = data_dir + '/valid'
    # test_dir = data_dir + '/test'

    # load arguments
    in_arg = get_input_args_predict()

    # load the model and optimizer we already used
    model = load_checkpoint(in_arg.checkpoint)

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


    image_path = in_arg.input

    topk = in_arg.top_k

    probs, classes = predict(image_path, model, topk, device)

    print("Probabilities: ", probs)
    print("Classes: ", classes)

    if(in_arg.category_names != None): # we have a file path
        category_path = in_arg.category_names
        flowers = category_mapper(probs, classes, category_path)
    
    
# END MAIN
    

if __name__ == "__main__":
    main()