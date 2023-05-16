import numpy as np
import matplotlib.pyplot as plt

from datasets.utils import generate_derived_data


def oracle(X, radius=1, mode=0):
    n = X.shape[0]
    y = np.zeros(n)
    for i in range(n):
        x = X[i, :]
        if np.linalg.norm(x) <= radius:
            y[i] = int(2 * mode)
        else:
            y[i] = int(2 * mode + 1)
    return y


def generate_2d_data(means, std, samples_per_cluster):
    n_centers=len(means)
    X_list=[]
    y_list=[]
    indicator_list=[]
    for i in range(n_centers):
        cov=np.diag([std**2]*2)
        X_temp=np.random.multivariate_normal(means[i], cov, size=samples_per_cluster)
        X_list.append(X_temp)
        y_list.append([i]*samples_per_cluster)
        indicator_list.append([int(i/2)]*samples_per_cluster)
    X=np.concatenate(X_list)
    y=np.concatenate(y_list)
    indicator=np.concatenate(indicator_list)
    return X, y, indicator


def generate_toy_data(random_state=0, samples_per_cluster=100):
    np.random.seed(random_state)

    means = [[0.5, np.sqrt(3) / 2],
             [np.sqrt(2) / 2, np.sqrt(2) / 2],
             [np.cos(np.pi/8), np.sin(np.pi/8)],
             [np.cos(np.pi/8), -np.sin(np.pi/8)],
             [0.5, -np.sqrt(3) / 2],
             [np.sqrt(2) / 2, -np.sqrt(2) / 2],
             ]
    std = 0.05
    X, y, indicator = generate_2d_data(means, std, samples_per_cluster)

    transfer_matrices = []
    for i in range(3):
        transfer_mat = np.random.randn(2, 2)
        transfer_matrices.append(transfer_mat)

    X_list = generate_derived_data(X, transfer_matrices)

    return X, X_list, y, indicator


from tools.plot import scatter_with_legends
if __name__=="__main__":
    X, X_list, y, indicator=generate_toy_data(random_state=7)

    plt.figure(figsize=[4,4])
    plt.xlim([-1,1])
    plt.ylim([-1, 1])
    plt.scatter(X[:,0], X[:,1], c=indicator)

    plt.figure(figsize=[4, 4])
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.scatter(X[:, 0], X[:, 1], c=y)

    for i in range(3):
        scatter_with_legends(X_list[i],indicator)

    plt.show()