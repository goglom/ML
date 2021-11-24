from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import dill


def get_wrong_pred(y_pred: np.ndarray, y_gt: np.ndarray, x: np.ndarray):
    mask = y_pred != y_gt
    return x[mask], y_pred[mask]


def show_data(x: np.ndarray, y=None, number: int=5, shape=None, dpi=80, highlight_pixeles=None, **kwargs):
    if shape is None:
        shape = (1, number)
    
    size = int(np.sqrt(x.shape[1]))
    images = x.reshape((x.shape[0], size, size))
    plt.figure(figsize=(3 * shape[1], 3 * shape[0]), dpi=dpi)
    for i, img in enumerate(images[:number]):
        plt.subplot(shape[0], shape[1], 1 + i)
        if not y is None:
            plt.title(f"Prediction: {y[i]}")
        plt.imshow(img, cmap=cm.gray_r)
        if not highlight_pixeles is None:
            xy_highlight = np.unravel_index(highlight_pixeles, img.shape)
            plt.scatter(*xy_highlight, **kwargs)

        plt.yticks([])
        plt.xticks([])
    plt.tight_layout()

def serialize(object, filename):
    with open("./tmp/" + filename + ".pickle", "wb") as file:
        dill.dump(object, file, protocol=dill.HIGHEST_PROTOCOL, fix_imports=True)


def deserialize(filename):
    with open("./tmp/" + filename + ".pickle", "rb") as file:
        return dill.load(file)
