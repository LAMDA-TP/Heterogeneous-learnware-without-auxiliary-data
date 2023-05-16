from collections import Counter
import numpy as np
from sklearn.preprocessing import MaxAbsScaler


def transform_labels(y, verbose=0):
    if verbose >= 1:
        print('original labels', Counter(y))

    label_list = [item[0] for item in Counter(y).most_common()]
    mapping = dict()
    for i in range(len(label_list)):
        mapping[label_list[i]] = i
    y_trans = [mapping[item] for item in y]

    if verbose >= 1:
        print('new labels', Counter(y_trans))
    return np.array(y_trans)


def generate_derived_data(data_2d, transfer_matrices):
    X_list=[]
    for i in range(len(transfer_matrices)):
        X_temp=data_2d @ transfer_matrices[i]
        scaler=MaxAbsScaler()
        X_temp=scaler.fit_transform(X_temp)
        X_list.append(X_temp)
    return X_list