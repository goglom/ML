import numpy as np
from sklearn.tree import DecisionTreeClassifier


class RandomForestClassifier:
    def __init__(self, n_trees: int, subset_size: float=0.5, random_state: int=0) -> None:
        self.classifiers = [DecisionTreeClassifier(criterion='gini', max_features='sqrt') for _ in range(n_trees)]
        self.subset_size = subset_size
        self.random_state = random_state

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        gen = np.random.RandomState(self.random_state)
        subset_size = int(x.shape[0] * self.subset_size)

        for cls in self.classifiers:
            subset_indices = gen.choice(x.shape[0], subset_size)
            x_subset = x[subset_indices, ...]
            y_subset = y[subset_indices, ...]
            cls.fit(x_subset, y_subset)

    def predict(self, x) -> np.ndarray:
        pred_table = np.zeros((x.shape[0], len(self.classifiers)), dtype=np.int64)
        for i, cls in enumerate(self.classifiers):
            pred_table[..., i] = cls.predict(x)
        result_pred = np.zeros((x.shape[0]), dtype=np.int64)

        for i, pred in enumerate(pred_table):
            clases, counts = np.unique(pred, return_counts=True)
            result_pred[i] = clases[np.argmax(counts)]

        return result_pred

        


if __name__ == "__main__":
    gen = np.random.RandomState(0)
    x = np.array([1, 1,1, 2, 3, 4, 4, 4, 5, 0, 1])

    print(gen.choice(x.shape[0], 4))
    print(gen.choice(x.shape[0], 4))
    print(gen.choice(x.shape[0], 4))
    