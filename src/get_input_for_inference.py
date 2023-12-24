#!/usr/bin/env python3

import argparse


def get_input_for_inference():
    """
    Retrieves and parses the flags
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_path",
        default="flowers/test/1/image_06743.jpg",
        help="The path to the image we want to inference on",
    )
    parser.add_argument(
        "checkpoint",
        default="save_directory/barak_model.pth",
        help="The path to the model we want to use",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Print the top_k best inference according to the model",
    )
    parser.add_argument(
        "--category_names",
        type=str,
        default="cat_to_name.json",
        help="Choose a dictionary , categories to real names",
    )
    parser.add_argument(
        "--gpu", action=argparse.BooleanOptionalAction, help="Apply GPU"
    )
    return parser.parse_args()
