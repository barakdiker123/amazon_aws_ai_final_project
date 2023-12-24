import argparse


def get_input_args():
    """
    Retrieves and parses the 3 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's
    argparse module to created and defined these 3 command line arguments. If
    the user fails to provide some or all of the 3 arguments, then the default
    values are used for the missing arguments.
    Command Line Arguments:
      1. Image Folder as --dir with default value 'pet_images'
      2. CNN Model Architecture as --arch with default value 'vgg'
      3. Text File with Dog Names as --dogfile with default value 'dognames.txt'
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_directory", help="Path to the folder of images to train on"
    )
    parser.add_argument(
        "--gpu", action=argparse.BooleanOptionalAction, help="Apply GPU"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="vgg13",
        help="Choose Feature model for example vgg",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="save_directory",
        help="Where to save the model save_directory",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="The learning rate of the training classifier ",
    )
    parser.add_argument(
        "--hidden_units",
        type=int,
        default=512,
        help="Hidden layer of the classifier ",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs to train the model",
    )
    # Replace None with parser.parse_args() parsed argument collection that
    # you created with this function
    return parser.parse_args()


if __name__ == "__main__":
    my_input = get_input_args()
    print(my_input.data_directory)
    if my_input.gpu:
        print("GPU Works !")
