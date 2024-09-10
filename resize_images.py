import cv2
import numpy as np
import logging
import os
from typing import Tuple
import argparse
import yaml
import sys

def load_resize_and_save_image(
    image_path: str, 
    target_path: str,
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Load an image, resize it while maintaining the aspect ratio, and save it with the same name.
    
    Args:
        image_path (str): Path to the image file.
        target_path (str): Path to which the resized image file will be saved.
        target_size (Tuple[int, int]): Target size for resizing (width, height).
    
    Raises:
        ValueError: If the image cannot be loaded.
        IOError: If the image cannot be saved.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        logging.error(f"Unable to load image: {image_path}")
        return 0
        
    original_height, original_width = img.shape[:2]
    target_width, target_height = target_size

    # Calculate aspect ratio of the original and target
    aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height
    
    # Determine new dimensions while maintaining the aspect ratio
    if aspect_ratio > target_aspect_ratio:
        # Resize based on width
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
    else:
        # Resize based on height
        new_height = target_height
        new_width = int(new_height * aspect_ratio)
    
    logging.info(f"Resizing image from {img.shape[:2]} to {new_width}x{new_height}")
    
    # Resize the image
    img_resized = cv2.resize(img, (new_width, new_height))
    
    # Construct the save path with the same name
    _, filename = os.path.split(image_path)
    basename, ext = os.path.splitext(filename)
    save_path = os.path.join(target_path, f"{basename}{ext}")
    
    # Save the resized image
    if not cv2.imwrite(save_path, img_resized):
        logging.error(f"Unable to save image to: {save_path}")
        raise IOError(f"Unable to save image to: {save_path}")
    
    logging.info(f"Image successfully resized and saved to: {save_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize the downloaded images to same size.")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file {args.config} not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file {args.config}: {e}")
        sys.exit(1)

    original_image_directory = config["original_image_directory"]
    target_image_directory = config["image_directory"]
    os.makedirs(target_image_directory, exist_ok=True)
    for file in os.listdir(original_image_directory):
        image_path = os.path.join(original_image_directory, file)
        load_resize_and_save_image(image_path, target_image_directory,(640, 480))