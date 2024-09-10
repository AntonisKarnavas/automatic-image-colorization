import argparse
import os
import sys
import cv2
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
from scipy.spatial import distance
import matplotlib.pyplot as plt
import yaml
import logging
import time


def get_gabor_features(superpixel: np.ndarray) -> np.ndarray:
    """
    Compute Gabor features for a given superpixel.

    Args:
        superpixel (np.ndarray): Flattened pixel values of the superpixel.

    Returns:
        np.ndarray: Mean Gabor features vector.
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
            filtered_img = cv2.filter2D(superpixel, cv2.CV_8UC3, kernel)
            gabor_vector.append(filtered_img.reshape(-1))
    return np.mean(np.transpose(np.array(gabor_vector)), axis=0)


def visualize_superpixels(
    image: np.ndarray,
    segments: np.ndarray,
    output_directory: str,
    img_filename: str,
    n_superpixels: int,
):
    """
    Visualize superpixels and save the image.

    Args:
        image (np.ndarray): Input image.
        segments (np.ndarray): Superpixel segments.
        img_filename (str): Filename to save the visualized image.
        n_superpixels (int): Number of superpixels.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(mark_boundaries(img_as_float(image), segments))
    plt.axis("off")
    os.makedirs(os.path.join(output_directory, "superpixeled_images"), exist_ok=True)
    plt.savefig(
        os.path.join(
            output_directory,
            "superpixeled_images",
            f"{img_filename}_divided_by_{n_superpixels}_superpixels.png",
        )
    )
    plt.close()


def main(config: dict):

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    num_superpixels = config["num_superpixels"]
    image_directory = config["image_directory"]
    output_directory = config["output_directory"]
    palette_sizes = config["palette_sizes"]

    logging.info(f"Superpixels to split images: {num_superpixels}")

    if not os.path.exists(output_directory):
        os.makedirs(os.path.join(output_directory, "superpixeled_images"))
        os.makedirs(os.path.join(output_directory, "color_palettes"))
        logging.info(f"Created '{output_directory}' directory to save data and plots.")

    image_files = [
        f
        for f in os.listdir(image_directory)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    logging.info(f"Found {len(image_files)} images.")

    start_time = time.time()

    for palette_size in palette_sizes:
        logging.info(f"Processing palette size: {palette_size}")
        palette = np.load(
            os.path.join(output_directory, "color_palettes", f"p{palette_size}.npz")
        )["arr_0"]

        for n_superpixels in num_superpixels:
            logging.info(f"Number of superpixels: {n_superpixels}")

            sift_gabor_vectors = []
            color_labels = []

            for image_filename in image_files:
                logging.info(f"Processing image: {image_filename}")
                image = cv2.cvtColor(
                    cv2.imread(os.path.join(image_directory, image_filename)),
                    cv2.COLOR_BGR2RGB,
                )

                # Split it in superpixels
                segments = slic(
                    img_as_float(image),
                    compactness=0.02,
                    n_segments=n_superpixels,
                    sigma=5,
                )
                visualize_superpixels(
                    image, segments, output_directory, image_filename, n_superpixels
                )

                # Create sift object
                sift = cv2.SIFT_create()

                # Convert image to grayscale so we can extract the sift and gabor features
                image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                # Convert image to Lab colour space, to select in which colour class of our palette belongs
                image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

                # Get dimensions of image (doesnt matter the color space in which is depicted)
                height, width, _ = image_lab.shape

                # Iterate all superpixels
                for seg_val in np.unique(segments):

                    # Create a mask with the value of 255 in the coordinates of the pixels of the currently examined superpixel
                    mask = np.zeros(image_lab.shape[:2], dtype="uint8")
                    mask[segments == seg_val] = 255

                    # Get the [a,b] values of Lab color space in our currently examined superpixel
                    superpixel_pixels = [
                        image_lab[x, y][1:]
                        for x in range(height)
                        for y in range(width)
                        if mask[x, y] == 255
                    ]

                    # Compute the mean of all the [a,b] colour arrays of the currently examined superpixel
                    # and find the closest one in our colour palette using euclidean distance.
                    # Then add that colour class in the labels vector
                    superpixel_mean = np.array(superpixel_pixels).mean(axis=0)
                    distances = [
                        distance.euclidean(superpixel_mean, color) for color in palette
                    ]
                    color_labels.append(palette[np.argmin(distances)])

                    # Get the grayscaled version of the currently examined superpixel
                    gray_superpixel = cv2.bitwise_and(image_gray, image_gray, mask=mask)

                    # Compute the sift feature vector/vectors from the superpixel
                    _, descriptors = sift.detectAndCompute(gray_superpixel, None)
                    sift_vector = (
                        descriptors.mean(axis=0)
                        if descriptors is not None
                        else np.zeros(128, dtype=int)
                    )

                    # Extract only the pixels of the superpixel, using the mask, in a flatten structure
                    flat_mask = mask.reshape(-1)
                    flat_img = gray_superpixel.reshape(-1)
                    extract_superpixel = flat_img[flat_mask == 255]

                    # Compute-extract the gabor features from each superpixel
                    gabor_vector = get_gabor_features(extract_superpixel)

                    # Create a 168 vector of features for each superpixel
                    sift_gabor_vectors.append(
                        np.concatenate((sift_vector, gabor_vector))
                    )

            os.makedirs(os.path.join(output_directory, "features"), exist_ok=True)
            os.makedirs(os.path.join(output_directory, "labels"), exist_ok=True)

            np.savez_compressed(
                os.path.join(
                    output_directory,
                    "features",
                    f"c{palette_size}_s{n_superpixels}.npz",
                ),
                np.array(sift_gabor_vectors),
            )
            np.savez_compressed(
                os.path.join(
                    output_directory, "labels", f"c{palette_size}_s{n_superpixels}.npz"
                ),
                np.array(color_labels),
            )

    end_time = time.time()
    logging.info(
        f"All images split into superpixels successfully. Total execution time: {end_time - start_time:.2f} seconds."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate features and labels from images using superpixels and color palettes."
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
