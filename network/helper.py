import matplotlib.pyplot as plt
import os
import random
import numpy as np
import tensorflow as tf
from pathlib import Path


def add_subplot(image, n_cols, n_rows, index):
    plot_obj = plt.subplot(n_cols, n_rows, index)
    plt.imshow(image, cmap="binary")
    plt.axis("off")
    return plot_obj


def save_plot_as_image(path, filename, dpi=300):
    Path(path).mkdir(parents=True, exist_ok=True)
    full_path = f"{path}/{filename}"
    plt.savefig(full_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()


def set_seeds(value):
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ["PYTHONHASHSEED"] = str(value)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(value)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(value)
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(value)
