import sys
import os
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import logging
import argparse
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple
import matplotlib.pyplot as plt
from skimage import color
import time
import multiprocessing


def load_resize_and_convert_image(
    image_path: str, output_directory:str, target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Load an image, resize it, and convert it to LAB color space.

    Args:
        image_path (str): Path to the image file.
        output_directory (str): Target path to where the lab images will be saved.
        target_size (Tuple[int, int]): Target size for resizing.

    Returns:
        np.ndarray: Resized image in LAB color space.
    """
    img = cv2.imread(image_path)
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    if img is None:
        raise ValueError(f"Unable to load image: {image_path}")
    img_resized = cv2.resize(img, target_size)
    
    os.makedirs(os.path.join(output_directory, 'lab_images'), exist_ok=True)
    # Construct the save path with the same name
    _, filename = os.path.split(image_path)
    cv2.imwrite(os.path.join(output_directory, 'lab_images', filename), lab_image)
    
    return cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)


def generate_palette(size: int, ab_data: np.ndarray) -> np.ndarray:
    """
    Generate a color palette using K-means clustering.

    Args:
        size (int): Number of colors in the palette.
        ab_data (np.ndarray): A and B channel data from LAB color space.

    Returns:
        np.ndarray: Sorted cluster centers representing the color palette.
    """
    kmeans = MiniBatchKMeans(
        n_clusters=size,
        init="k-means++",
        n_init=10,
        max_iter=200,
        tol=0.01,
        batch_size=2048,
        verbose=0,
        random_state=42,
    )
    kmeans.fit(ab_data)
    return kmeans.cluster_centers_


def _find_closest_lab(array_lab: np.ndarray, array_ab: np.ndarray) -> np.ndarray:
    """
    Given an array of (l, a, b) values and a smaller array of (a, b) values,
    this function finds the closest (l, a, b) point for each (a, b) pair based on Euclidean distance.

    Args:
    array_lab (numpy.ndarray): A numpy array of shape (N, 3) representing (l, a, b) points, where N is the number of points.
    array_ab (numpy.ndarray): A numpy array of shape (M, 2) representing (a, b) pairs, where M is the number of points.

    Returns:
    numpy.ndarray: A numpy array of shape (M, 3) where each row is the (l, a, b) point closest to the corresponding (a, b) pair.
    """

    closest_lab = np.zeros((array_ab.shape[0], 3))
    for i, (a_target, b_target) in enumerate(array_ab):
        a_values = array_lab[:, 1]
        b_values = array_lab[:, 2]

        distances = np.sqrt((a_values - a_target) ** 2 + (b_values - b_target) ** 2)
        closest_idx = np.argmin(distances)
        closest_lab[i] = array_lab[closest_idx]

    return closest_lab


def visualize_palette(
    palette: np.ndarray, data: np.ndarray, size: int, output_dir: str
):
    """
    Visualize the generated color palette.

    Args:
        palette (np.ndarray): The generated color palette in LAB color space.
        size (int): Number of colors in the palette.
        output_dir (str): Directory to save the visualization.
    """

    closest_lab = _find_closest_lab(data, palette)
    _, ax = plt.subplots(figsize=(10, 2))
    for i, color in enumerate(closest_lab):
        color = cv2.cvtColor(
            color.reshape(1, 1, 3).astype(np.float32), cv2.COLOR_Lab2BGR
        )
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color[0][0]))
    ax.set_xlim(0, size)
    ax.set_ylim(0, 1)
    ax.axis("off")
    os.makedirs(output_dir, exist_ok=True)
    plt.title(f"Generated Palette with {size} Colors")
    plt.savefig(os.path.join(output_dir, f"palette_{size}_colors.png"))
    plt.close()


def main(config: dict):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    palette_sizes = config["palette_sizes"]
    image_directory = config["image_directory"]
    output_directory = config["output_directory"]
    num_workers = config.get("num_workers", multiprocessing.cpu_count())
    target_size = tuple(config.get("target_size", (640, 480)))

    logging.info(f"Palette sizes to generate: {palette_sizes}")
    logging.info(f"Using {num_workers} workers")
    logging.info(f"Resizing images to {target_size}")

    if not os.path.exists(output_directory):
        os.makedirs(os.path.join(output_directory, "palette_plots"))
        os.makedirs(os.path.join(output_directory, "color_palettes"))
        logging.info(f"Created '{output_directory}' directory to save data and plots.")

    image_files = [
        f
        for f in os.listdir(image_directory)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    logging.info(f"Found {len(image_files)} images.")

    start_time = time.time()

    logging.info("Loading, resizing, and converting images to LAB color space...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                load_resize_and_convert_image,
                os.path.join(image_directory, img),
                output_directory,
                target_size,
            )
            for img in image_files
        ]
        images = [
            future.result()
            for future in tqdm(
                as_completed(futures), total=len(image_files), desc="Processing images"
            )
        ]

    logging.info("Extracting and flattening l, a and b channels...")
    ab_data = np.vstack([img[:, :, 1:].reshape(-1, 2) for img in images])
    lab_data = np.vstack([img[:, :, :].reshape(-1, 3) for img in images])

    logging.info("Generating color palettes...")
    for size in tqdm(palette_sizes, desc="Generating palettes"):
        logging.info(f"Generating palette with {size} colors...")
        palette = generate_palette(size, ab_data)
        os.makedirs(os.path.join(output_directory, "color_palettes"), exist_ok=True)
        np.savez_compressed(
            os.path.join(output_directory, "color_palettes", f"p{size}.npz"), palette
        )
        visualize_palette(
            palette, lab_data, size, os.path.join(output_directory, "palette_plots")
        )
        logging.info(f"Palette with {size} colors saved and visualized.")

    end_time = time.time()
    logging.info(
        f"All palettes generated successfully. Total execution time: {end_time - start_time:.2f} seconds."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate color palettes from images.")
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
