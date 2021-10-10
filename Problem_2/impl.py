import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from itertools import cycle, product
from sklearn.metrics import accuracy_score


def grid(size):
    return product(range(size), range(size))


def plot_features_data(X: np.array, Y: np.array, f_names):
    num_features = X.shape[1]
    print(X.shape)

    for i, (f1, f2) in enumerate(grid(num_features)):
        plt.subplot(num_features, num_features, 1 + i)

        if f1 == f2:
            plt.text(0.25, 0.5, f_names[f1])
        else:
            plt.scatter(X[:, f1], X[:, f2], c=Y)
            plt.xlabel(f_names[f1])
            plt.ylabel(f_names[f2])

    plt.tight_layout()


def plot_points(x: np.array, y: np.array, Classes: np.array, marked_indexes=None, axes_names:tuple[str, str] =None):
    plt.scatter(x, y, c=Classes)
    if axes_names is None:
        axes_names = ("x", "y")

    if not marked_indexes is None:
        plt.scatter(x[marked_indexes], y[marked_indexes], c='r', marker='x')

    plt.xlabel(axes_names[0])
    plt.ylabel(axes_names[1])    

class PotentialFunctionClassifier:
    def __init__(self, window_size: float) -> None:
        self.train_x = None
        self.charges = None
        self.indexes = None
        self.widnow_size = window_size
        self.train_y = None
        self.classes = None
        self.kernel = lambda x: 1 / (x + 1)

    def _remeber(self, train_x: np.array, train_y: np.array) -> None:
        self.train_x = train_x
        self.classes = np.unique(train_y)
        self.charges = np.zeros_like(train_y, dtype=int)
        self.indexes = np.arange(0, len(train_y), dtype=int)
        self.train_y = train_y

    def _reduce(self) -> None:
        non_zero_mask = self.charges != 0.0

        self.train_x = self.train_x[non_zero_mask, ...]
        self.train_y = self.train_y[non_zero_mask, ...]
        self.charges = self.charges[non_zero_mask, ...]
        self.indexes = self.indexes[non_zero_mask, ...]

    def fit(self, train_x: np.array, train_y: np.array, epochs: int=1) -> None:
        assert train_x.shape[0] == train_y.shape[0]

        self._remeber(train_x, train_y)
        error = np.inf
        self.charges[0] = 1.

        for _ in range(epochs):
            for i in range(self.train_x.shape[0]):
                if self.predict(self.train_x[i]) != self.train_y[i]:
                    self.charges[i] += 1


        self._reduce()

    def predict(self, x: np.array):
        test_x = np.copy(x)

        if len(test_x.shape) < 2:
             test_x = test_x[np.newaxis, :]

        diff = test_x[:, np.newaxis, :] - self.train_x[np.newaxis, :, :]
        distances = np.sqrt(np.sum((diff**2), -1))
        weights = self.charges * self.kernel(distances / self.widnow_size)
        table = np.zeros((test_x.shape[0], len(self.classes)))

        for c in self.classes:
            table[:, c] = np.sum(weights[:, self.train_y == c], axis=1)

        return np.argmax(table, axis=1)



def _main():
    a = np.array([0, 1, 9, 2, 2, 2,2])
    print(a[np.array([0, 1, 2])])




if __name__ == '__main__':
    _main()