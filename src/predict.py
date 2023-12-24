#!/usr/bin/env python3

import torch
import torch.nn as nn
import json
from get_input_for_inference import get_input_for_inference
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def process_image(image):  # image is the path of the image ?
    """Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    """
    with Image.open(image) as im:  # where image is "hopper.jpg" for example
        pil_image = im
        # im.show()
        # im.thumbnail(256).show() # barak: "Can remove ? "
        # np_image = np.array(pil_image)
        test_transforms = transforms.Compose(
            [
                transforms.Resize(size=256),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return test_transforms(pil_image)
    print("BARAK")
    return None


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def predict(image_path, model, topk, device):
    """Predict the class (or classes) of an image using a trained deep learning model."""
    image_tensor = process_image(image_path)
    image_tensor = image_tensor.to(device)
    image_corped = transforms.CenterCrop(224)(image_tensor)
    image_corped = image_corped.view(1, 3, 224, 224)
    model.to(device)
    pred = model(image_corped)
    log_prob, indices = torch.topk(pred, topk)
    return (
        torch.exp(log_prob)[0].detach().cpu().numpy(),
        indices[0].detach().cpu().numpy(),
    )


def pretty_display(image_path, idx_to_class, model, topk, device):
    fig, ax = plt.subplots(2)
    img_display = process_image(image_path)
    imshow(img_display, ax=ax[0])

    log_prob, indices = predict(image_path, model, topk, device)
    indices_name = [cat_to_name[idx_to_class[indice_name]] for indice_name in indices]
    ax[0].set_title(indices_name[0])

    # Example data
    people = indices_name
    y_pos = np.arange(len(people))
    performance = log_prob

    ax[1].barh(y_pos, performance, align="center")
    ax[1].set_yticks(y_pos, labels=people)
    ax[1].invert_yaxis()  # labels read top-to-bottom
    # ax[1].set_xlabel('Performance')
    # ax[1].set_title('How fast do you want to go today?')

    plt.show()


if __name__ == "__main__":
    my_input = get_input_for_inference()
    with open(my_input.category_names, "r") as f:
        cat_to_name = json.load(f)
        # print(cat_to_name["21"])
    model = torch.load(my_input.checkpoint)
    if my_input.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()
    #prob, indices = predict(my_input.image_path, model, my_input.top_k, device)
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    #indices_name = [cat_to_name[idx_to_class[indice_name]] for indice_name in indices]
    pretty_display(my_input.image_path, idx_to_class, model, my_input.top_k, device)
