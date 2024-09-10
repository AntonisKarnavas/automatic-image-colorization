import argparse
import os
import sys
import pickle
import numpy as np
import logging
import yaml
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import time


def main(config: dict):
    """
    Train SVM models based on the provided configuration and save them to disk.

    Args:
        config (Dict): Configuration dictionary containing palette sizes, number of superpixels, and file paths.
    """
    num_superpixels = config["num_superpixels"]
    output_directory = config["output_directory"]
    palette_sizes = config["palette_sizes"]

    start_time = time.time()

    for palette_size in palette_sizes:
        logging.info(f"Processing palette size: {palette_size}")
        palette = np.load(
            os.path.join(output_directory, "color_palettes", f"p{palette_size}.npz")
        )["arr_0"]

        for n_superpixels in num_superpixels:
            logging.info(f"Number of superpixels: {n_superpixels}")

            try:
                features = np.load(
                    os.path.join(
                        output_directory,
                        "features",
                        f"c{palette_size}_s{n_superpixels}.npz",
                    )
                )["arr_0"]
                labels = np.load(
                    os.path.join(
                        output_directory,
                        "labels",
                        f"c{palette_size}_s{n_superpixels}.npz",
                    )
                )["arr_0"]
            except FileNotFoundError as e:
                logging.error(f"File not found: {e}")
                continue

            indices = []
            for label in labels:
                for idx, palette_color in enumerate(palette):
                    if np.array_equal(label, palette_color):
                        indices.append(idx)
                        break

            if not indices:
                logging.warning(
                    f"No matching labels found for palette size {palette_size} and superpixels {n_superpixels}."
                )
                continue
            param_grid = {
            'C': [0.1, 1, 10, 50, 100],
            'gamma': ['scale'],
            'kernel': ['linear', 'rbf', 'poly']
}            
            svc =  svm.SVC()
            
            X_train, X_test, y_train, y_test = train_test_split(features, indices, test_size=0.2, random_state=42)
            
            grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            
            os.makedirs(os.path.join(output_directory, "models"), exist_ok=True)
            model_path = os.path.join(
                output_directory,
                "models",
                f"model_{palette_size}_palette_and_{n_superpixels}_superpixels.pkl",
            )
            
            with open(model_path, "wb") as model_file:
                pickle.dump(best_model, model_file)
            logging.info(f"Model saved to {model_path}")
        
            report = classification_report(y_test, y_pred)
            logging.info("Classification Report:\n%s", report)

    end_time = time.time()
    logging.info(
        f"All models trained successfully. Total execution time: {end_time - start_time:.2f} seconds."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SVM models based on color palettes and superpixels."
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

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
