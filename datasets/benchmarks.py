"""
This file is used to load or generate datasets with multiple feature spaces.
"""

from copy import deepcopy
from collections import Counter
import numpy as np
import os

from mvlearn.datasets import load_UCImultifeature
from sklearn.datasets import load_digits, fetch_kddcup99, fetch_covtype
from sklearn.preprocessing import LabelBinarizer, MaxAbsScaler

from datasets.load_anuran_dataset import get_anuran_dataset
from datasets.utils import transform_labels, generate_derived_data


class DatasetLoader():

    def __init__(self, data_home='./benchmarks'):
        self.data_home=data_home

    def load_task_dataset(self, task_name , list_generation_type='split', dim_list=None, random_state=0):
        if task_name=='mfeat_1':
            X_list, y = load_dataset_multi_fea('mfeat', self.data_home)
            X_list=X_list[0:3]
        elif task_name=='mfeat_2':
            X_list, y = load_dataset_multi_fea('mfeat', self.data_home)
            X_list = X_list[3:6]
        elif task_name == 'digits' or 'kddcup99' or 'covtype' or 'anuran':
            X, y = load_dataset_single_fea(task_name, self.data_home)
            if list_generation_type == 'split':
                X_list = split_dataset_single_fea(X, random_state)
            else:
                if dim_list == None:
                    dim_list = [X.shape[1]] * 3
                X_list = transform_dataset_single_fea(X, dim_list, random_state)
        X_list = self._normaliza_X_list(X_list, MaxAbsScaler())
        return X_list, y

    def _normaliza_X_list(self, X_list, scaler):
        X_list_new=deepcopy(X_list)
        for i in range(len(X_list)):
            X_list_new[i]=scaler.fit_transform(X_list[i])
        return X_list_new


def load_dataset_multi_fea(dataset_name, data_home='./benchmarks', verbose=0):
    """
        load the dataset with multiple feature spaces.
    """
    if dataset_name == 'mfeat':
        X_list, y = load_UCImultifeature()
        y=np.array(y, dtype=int)
    return X_list, y


def load_dataset_single_fea(dataset_name, data_home='./benchmarks', verbose=0):
    """
        load the dataset with single feature space.
    """
    if dataset_name == 'digits':
        data = load_digits()
        X = data['data']
        y = data['target']
    elif dataset_name == 'kddcup99':
        data = fetch_kddcup99(data_home=data_home)
        X = data['data']
        y = data['target']
        y = transform_labels(y, verbose=verbose)
        idx = select_samples(y, 400, 1000)
        X = X[idx]
        y = y[idx]
        lb = LabelBinarizer()
        x1 = lb.fit_transform(X[:, 1].astype(str))
        x2 = lb.fit_transform(X[:, 2].astype(str))
        x3 = lb.fit_transform(X[:, 3].astype(str))
        X = np.c_[X[:, :1], x1, x2, x3, X[:, 4:]]
    elif dataset_name == 'covtype':
        data = fetch_covtype(data_home=data_home)
        X = data['data']
        y = data['target']
        y = transform_labels(y, verbose=verbose)
        idx = select_samples(y, 1000, 9000)
        X = X[idx]
        y = y[idx]
    elif dataset_name=='anuran':
        X, y= get_anuran_dataset(os.path.join(data_home, dataset_name))
    return X,y


def rearrange_list(X_list):
    dim_list=[]
    for i in range(len(X_list)):
        dim_list.append(X_list[i].shape[1])
    idx_list=np.argsort(dim_list)
    X_list_new=[]
    for idx in idx_list:
        X_list_new.append(X_list[idx])
    return X_list_new


def split_dataset_single_fea(X, random_state=0):
    np.random.seed(random_state)
    n_dim=X.shape[1]
    idx_list=np.array(list(range(n_dim)))
    np.random.shuffle(idx_list)
    idx_1=int(n_dim/3)
    idx_2=int(n_dim/3*2)
    X_list=[X[:,0:idx_1],X[:,idx_1:idx_2],X[:,idx_2:n_dim]]
    return X_list

def transform_dataset_single_fea(X, dim_list, random_state=0):
    np.random.seed(random_state)
    transfer_matrices=[]
    for i in range(len(dim_list)):
        transfer_mat = np.random.randn(X.shape[1], dim_list[i])
        transfer_matrices.append(transfer_mat)

    X_list = generate_derived_data(X, transfer_matrices)
    return X_list


def describe_dataset(X_list, y, dataset_name=''):
    print('INFO of dataset %s:'%dataset_name)
    for i in range(len(X_list)):
        print('shape of data in the %d-th feature space:'%i, X_list[i].shape)
    print('shape of labels:', y.shape)
    c=Counter(y)
    label_set=sorted(c)
    label_size_list=[]
    for label in label_set:
        label_size_list.append(c[label])
    print('labels are', label_set)
    print('size of each label is', label_size_list)


def select_samples(labels, n_per_class, th, random_seed=0):
    """

    :param labels: label set
    :param n_per_class:
    :param th: ignore the class whose cardinality is smaller than th
    :return:
    """
    pass
    n_classes = len(set(labels))
    idx_list = []
    for i in range(n_classes):
        temp_idx_list = np.argwhere(labels == i)
        if len(temp_idx_list)>1:
            temp_idx_list=temp_idx_list.squeeze()
        if len(temp_idx_list) >= th:
            rng = np.random.RandomState(random_seed)
            temp_idx_list = rng.choice(temp_idx_list, n_per_class, replace=False)
            idx_list.extend(temp_idx_list)
    return np.array(idx_list)


import warnings
warnings.filterwarnings('ignore')

if __name__=='__main__':
    task_name_list = ['mfeat_1', 'mfeat_2', 'digits', 'kddcup99', 'covtype',  'anuran']
    data_home='./benchmarks'
    dataset_loader=DatasetLoader(data_home)
    for task_name in task_name_list:
        X_list, y= dataset_loader.load_task_dataset(task_name, random_state=0)
        describe_dataset(X_list, y, task_name)
        print()