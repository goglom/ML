import numpy as np
import matplotlib.pyplot as plt
import integrals


def metrics(y_pred: np.array, y_gt: np.array):
    return {
        "TP": np.count_nonzero(np.logical_and(y_pred == 1, y_gt == 1)),
        "TN": np.count_nonzero(np.logical_and(y_pred == 0, y_gt == 0)),
        "FP": np.count_nonzero(np.logical_and(y_pred == 1, y_gt == 0)),
        "FN": np.count_nonzero(np.logical_and(y_pred == 0, y_gt == 1))
    }


def true_pos_rate(**kwargs):
    return kwargs["TP"] / (kwargs["TP"] + kwargs["FN"])


def false_pos_rate(**kwargs):
    return kwargs["FP"] / (kwargs["TN"] + kwargs["FP"])


def precision(**kwargs):
    return kwargs["TP"] / (kwargs["TP"] + kwargs["FP"])


def _get_curve_impl(y_prob: np.array, y_ground_truth: np.array, thresholds: np.array, x_func, y_func):
    x = np.zeros_like(thresholds)
    y = np.zeros_like(thresholds)
    for i, threshold in enumerate(sorted(thresholds, reverse=True)):
        y_predict = y_prob >= threshold
        met = metrics(y_predict, y_ground_truth)
        x[i] = x_func(**met)
        y[i] = y_func(**met)

    return x, y


def get_roc_curve(y_prob: np.array, y_ground_truth: np.array):
    """
    returns: FPR, TPR
    """
    thresholds = np.concatenate(([0.], y_prob, [1.]))
    return _get_curve_impl(y_prob, y_ground_truth, thresholds, false_pos_rate, true_pos_rate)


def get_pr_curve(y_prob: np.array, y_ground_truth: np.array):
    """
        returns: recall, precision
    """
    rec, pre = _get_curve_impl(y_prob, y_ground_truth, y_prob, true_pos_rate, precision)
    return np.concatenate(([0.], rec)), np.concatenate(([1.], pre))


def auc(x_vals, y_vals):
    return integrals.integral_trapeze(y_vals, x_vals)


def coin_flip(y_labels: np.array):
    rand_gen = np.random.RandomState(1)
    y_pred = rand_gen.random_sample(y_labels.shape[0])
    y_gt = rand_gen.random_sample(y_labels.shape[0]) - 0.5 >= 0

    return y_gt, y_pred


if __name__ == '__main__':
    y_preds = coin_flip(np.array([1, 0, 1, 0, 1, 0]))
    
    print(y_preds)
