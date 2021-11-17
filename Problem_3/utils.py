from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np


def get_wrong_pred(y_pred: np.ndarray, y_gt: np.ndarray, x: np.ndarray):
    mask = y_pred != y_gt
    return x[mask], y_pred[mask]


def show_data(x: np.ndarray, y=None, number: int=5, dpi=80, highlight_pixeles=None, **kwargs):
    size = int(np.sqrt(x.shape[1]))
    images = x.reshape((x.shape[0], size, size))
    plt.figure(figsize=(3 * number, 3), dpi=dpi)
    for i, img in enumerate(images[:number]):
        plt.subplot(1, number, 1 + i)
        if not y is None:
            plt.title(f"Prediction: {y[i]}")
        plt.imshow(img, cmap=cm.gray_r)
        if not highlight_pixeles is None:
            xy_highlight = np.unravel_index(highlight_pixeles, img.shape)
            plt.scatter(*xy_highlight, **kwargs)

        plt.yticks([])
        plt.xticks([])
    plt.tight_layout()
