import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from itertools import cycle, product
from sklearn.metrics import accuracy_score


def _grid(size):
    return product(range(size), range(size))


def plot_data(X: np.array, Y: np.array, f_names: list[str], figsize=(14, 8)):
    num_features = X.shape[1] - 1

    plt.figure(figsize=figsize)

    for i, (f1, f2) in enumerate(_grid(num_features)):
        plt.subplot(num_features, num_features, 1 + i)

        if f1 == f2:
            plt.text(0.25, 0.5, f_names[f1])
        else:
            plt.scatter(X[:, f1], X[:, f2], c=Y)
            plt.xlabel(f_names[f1])
            plt.ylabel(f_names[f2])


class PotentialFunctionClassifier:
    def __init__(self, window_size: float) -> None:
        self.train_x = None
        self.charges = None
        self.widnow_size = window_size
        self.train_y = None
        self.classes = None
        self.kernel = lambda x: 1 / (x + 1)

    def _remeber(self, train_x: np.array, train_y: np.array) -> None:
        self.train_x = train_x
        self.classes = np.unique(train_y)
        self.charges = np.zeros_like(train_y, dtype=int)
        self.train_y = train_y

    def _reduce(self) -> None:
        non_zero_mask = self.charges != 0.0

        self.train_x = self.train_x[non_zero_mask,...]
        self.train_y = self.train_y[non_zero_mask,...]
        self.charges = self.charges[non_zero_mask,...]

    def fit(self, train_x: np.array, train_y: np.array, error_threshold = 0.1) -> None:
        assert train_x.shape[0] == train_y.shape[0]

        self._remeber(train_x, train_y)
        error = np.inf
        self.charges[0] = 1.

        for i in cycle(range(self.train_x.shape[0])):
            if error < error_threshold:
                break

            pred_i = self.predict(self.train_x[i])

            if pred_i != self.train_y[i]:
                self.charges[i] += 1
            
            pred = self.predict(self.train_x)
            error = 1.0 - accuracy_score(self.train_y, pred)

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
    iris_ds = datasets.load_iris()
    X = iris_ds.data
    Y = iris_ds.target
    plot_data(X, Y, iris_ds.feature_names)


    classifier = PotentialFunctionClassifier(0.1)

    classifier.fit(X, Y)

    print(classifier.charges)

    print(accuracy_score(Y, classifier.predict(X)))
    
    plt.tight_layout()
    plt.show()
    
        




if __name__ == '__main__':
    _main()