import argparse
import os
import pickle
import numpy as np
import cv2
import logging
import yaml
from skimage.segmentation import slic
from skimage.util import img_as_float
import sys


def get_gabor(extract_superpixel: np.ndarray) -> np.ndarray:
    """
    Extract Gabor features from a superpixel.

    Args:
        extract_superpixel (np.ndarray): Flattened pixel values of the superpixel.

    Returns:
        np.ndarray: Gabor feature vector.
    """
    gabor_vector = []
    ksize = 10
    phi = 0
    sigma = 20
    gamma = 0.25
    for lamda in np.arange(0, np.pi, np.pi / 5):
        for theta in np.arange(0, np.pi, np.pi / 8):
            kernel = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F
            )
            fimg = cv2.filter2D(extract_superpixel, cv2.CV_8UC3, kernel)
            filtered_img = fimg.reshape(-1)
            gabor_vector.append(filtered_img)
    gabor_vector = np.mean(np.transpose(np.array(gabor_vector)), axis=0)
    return gabor_vector


def main(config: dict):
    """
    Process images, extract features, and classify using pre-trained models.

    Args:
        config (Dict): Configuration dictionary containing palette sizes, number of superpixels, and file paths.
    """

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    palette_sizes = config["palette_sizes"]
    output_directory = config["output_directory"]
    num_superpixels = config["num_superpixels"]
    test_image_directory = config["test_image_directory"]

    os.makedirs(os.path.join(output_directory, "results"))

    image_files = [
        f
        for _, _, filenames in os.walk(test_image_directory)
        for f in filenames
        if f.endswith((".png", ".jpg", ".jpeg"))
    ]

    for img_filename in image_files:
        logging.info(f"Processing image: {img_filename}")
        for palette_size in palette_sizes:
            palette = np.load(
                os.path.join(output_directory, "color_palettes", f"p{palette_size}.npz")
            )["arr_0"]
            for n_superpixels in num_superpixels:
                model_path = os.path.join(
                    output_directory,
                    "models",
                    f"model_{palette_size}_palette_and_{n_superpixels}_superpixels.pkl",
                )
                if not os.path.isfile(model_path):
                    logging.warning(f"Model file {model_path} not found. Skipping...")
                    continue

                logging.info(f"Loading model from {model_path}")
                loaded_model = pickle.load(open(model_path, "rb"))

                image_path = os.path.join(test_image_directory, img_filename)
                image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2LAB)[:, :, 0]
                sift_gabor_vec = []

                # Split it in superpixels and show the image
                segments = slic(
                    img_as_float(image),
                    compactness=0.02,
                    n_segments=n_superpixels,
                    sigma=5,
                    channel_axis=None,
                )

                # Create sift object
                sift = cv2.SIFT_create()

                # Iterate all superpixels
                for i, segVal in enumerate(np.unique(segments)):
                    # Create a mask with the value of 255 in the coordinates of the pixels of the currently examined superpixel
                    mask = np.zeros(image.shape[:2], dtype="uint8")
                    mask[segments == segVal] = 255

                    # Get the grayscaled version of the currently examined superpixel
                    graycurrsuperpixel = cv2.bitwise_and(image, image, mask=mask)

                    # Compute the sift feature vector/vectors from the superpixel
                    _, descriptors = sift.detectAndCompute(graycurrsuperpixel, None)
                    if descriptors is None:
                        sift_vec = np.zeros((128,), dtype=int)
                    else:
                        sift_vec = descriptors.mean(axis=0)

                    # Extract only the pixels of the superpixel, using the mask, in a flatten structure
                    flat_mask = mask.reshape(-1)
                    flat_img = graycurrsuperpixel.reshape(-1)
                    extract_superpixel = flat_img[flat_mask == 255]

                    # Compute-extract the gabor features from each superpixel
                    gabor_vec = get_gabor(extract_superpixel)

                    # Create a 168 vector of features for each superpixel
                    sift_gabor_vec.append(
                        np.concatenate((sift_vec, gabor_vec), axis=None)
                    )

                if not sift_gabor_vec:
                    logging.warning(
                        f"No superpixels found for image {img_filename} with palette {palette_size} and superpixels {n_superpixels}."
                    )
                    continue

                result = loaded_model.predict(sift_gabor_vec)
                new_img = np.zeros((480, 640, 2), dtype="uint8")

                for i, segVal in enumerate(np.unique(segments)):
                    new_img[segments == segVal] = palette[result[i]]

                output_img_path = os.path.join(
                    output_directory,
                    "results",
                    f"{img_filename[:-4]}_palette{palette_size}_superpixels{n_superpixels}.png",
                )
                cv2.imwrite(
                    output_img_path,
                    cv2.cvtColor(np.dstack((image, new_img)), cv2.COLOR_LAB2BGR),
                )
                logging.info(f"Result saved to {output_img_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process images and classify using pre-trained models."
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file {args.config} not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file {args.config}: {e}")
        sys.exit(1)

    main(config)
