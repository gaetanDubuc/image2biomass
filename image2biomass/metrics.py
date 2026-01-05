import numpy as np


def weighted_r2(y_true, y_pred, w):
    w_sum = np.sum(w)
    y_w_mean = np.sum(w * y_true) / w_sum

    ss_res = np.sum(w * (y_true - y_pred) ** 2)
    ss_tot = np.sum(w * (y_true - y_w_mean) ** 2)

    return 1 - (ss_res / ss_tot)
