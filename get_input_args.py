# Udacity Command Line App - Image Classifier
# Jared Teller
# 1/29/2021

# argparse
import argparse

# TODO 1: Define get_input_args function below please be certain to replace None
#       in the return statement with parser.parse_args() parsed argument 
#       collection that you created with this function
# 
def get_input_args_train():
    """
    Retrieves and parses the command line arguments.
    Command Line Arguments:
      1. Data Directory, Not Optional
      2. CNN Model Architecture as --arch with default value 'vgg13'
      3. Save Directory as save_dir
      4. Hyperparameters - Learning Rate
      5. Hyperparameters - Hidden Units
      6. Hyperparameters - epochs
      7. GPU
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """

    # Create the arg parser object
    parser = argparse.ArgumentParser()

    # Creating arg1, data_dir
    parser.add_argument('--dir', type=str, default='flowers', help='path to the folder of flower images, default is flowers')
    
    # Creating arg2, CNN Model Architecture
    parser.add_argument('--arch', type=str, default='vgg11', help='CNN Model Architecture to use. Options are: vgg11, vgg13, vgg16')

    # Creating arg3, Save Directory for checkpoints
    parser.add_argument('--save_dir', type=str, default=None, help='Save directory for checkpoints, default is current directory')
    
    # Creating arg3, learning rate
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for training the neural network')
    
    # Creating arg4, hidden units
    parser.add_argument('--hidden_units', type=int, default=4096, help='number of hidden units for the neural network')

    # Creating arg5, epochs
    parser.add_argument('--epochs', type=int, default=5, help='number of training epochs')
    
    # Creating arg6, GPU
    parser.add_argument('--gpu', action="store_true", help='enable GPU for training the NN')

    # Store the parsed arguments
    in_args = parser.parse_args()

    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return in_args

def get_input_args_predict():
    """
    Retrieves and parses the command line arguments.
    Command Line Arguments:
        1. path to image file - not optional
        2. checkpoint
        3. --top_k #
        4. --category_names file_name.json
        5. GPU
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """

    # Create the arg parser object
    parser = argparse.ArgumentParser()

    # Creating arg1, file
    parser.add_argument('input', type=str, default=None, help='path to the flower image you want the model to process. NEEDS TO BE RELATIVE e.g. ./flowers/test/1/img_001.jpg')
    
    # Creating arg2, checkpoint
    parser.add_argument('checkpoint', type=str, default=None, help='path to the checkpoint')

    # Creating arg3, top K
    parser.add_argument('--top_k', type=int, default=1, help='top_k most likely image classes')
    
    # Creating arg3, learning rate
    parser.add_argument('--category_names', type=str, default=None, help='category name mapping json file location')

    # Creating arg6, GPU
    parser.add_argument('--gpu', action="store_true", help='enable GPU for training the NN')

    # Store the parsed arguments
    in_args = parser.parse_args()

    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return in_args