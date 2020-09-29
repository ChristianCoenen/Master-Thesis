import matplotlib.pyplot as plt
from pathlib import Path


def add_subplot(image, n_cols, n_rows, index):
    plot_obj = plt.subplot(n_cols, n_rows, index)
    plt.imshow(image, cmap="binary")
    plt.axis("off")
    return plot_obj


def save_plot_as_image(path, filename, dpi=300):
    Path(path).mkdir(parents=True, exist_ok=True)
    full_path = f"{path}/{filename}"
    plt.savefig(full_path, dpi=dpi)
    plt.close()
