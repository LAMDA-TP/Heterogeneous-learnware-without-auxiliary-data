from cvxopt import solvers, matrix
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


def mmd_distance(X, Y, gamma=1, alpha=None, beta=None):
    """

    :param X: n_x*d
    :param Y: n_y*d
    :param gamma:
    :param alpha:
    :param beta:
    :return:
    """
    n_x=X.shape[0]
    n_y=Y.shape[0]
    if alpha is None:
        alpha=np.ones(n_x)/n_x
    if beta is None:
        beta=np.ones(n_y)/n_y

    alpha=np.array(alpha)
    beta=np.array(beta)

    x1 = rbf_kernel(X, X, gamma=gamma)
    x2 = rbf_kernel(Y, Y, gamma=gamma)
    x3 = rbf_kernel(X, Y, gamma=gamma)
    x1 = alpha.T @ x1 @ alpha
    x2 = beta.T @ x2 @ beta
    x3 = alpha.T @ x3 @ beta

    return x1 + x2 - 2 * x3


def mmd_distance_list(list_of_Phi,X,weights=None):
    gamma = list_of_Phi[0].gamma
    mmd_distances = []
    for i in range(len(list_of_Phi)):
        distance = mmd_distance(list_of_Phi[i].z, X, gamma=gamma,
                                alpha=list_of_Phi[i].beta, beta=weights)
        mmd_distances.append(distance)
    return mmd_distances


def coefficient_estimation(list_of_Phi,X,weights=None,solver='cvxopt', eq_constraint=True, neq_constraint=True):
    """

    :param list_of_Phi:
    :param X:
    :param weights:
    :param solver: {'cvxopt', 'inv'}
    :return:
    """
    c = len(list_of_Phi)
    gamma = list_of_Phi[0].gamma

    C = np.zeros(c)
    N = X.shape[0]
    if weights is None:
        weights = np.array([1 / N] * N)
    for i, Phi in enumerate(list_of_Phi):
        temp_array = Phi.eval(X)
        C[i] = np.dot(temp_array, weights)
    C = C.reshape(-1, 1)
    H = np.zeros((c, c))
    for i in range(c):
        for j in range(i, c):
            Z_i = np.array(list_of_Phi[i].z)
            Z_j = np.array(list_of_Phi[j].z)
            beta_i = np.array(list_of_Phi[i].beta).reshape(-1, 1)
            beta_j = np.array(list_of_Phi[j].beta).reshape(-1, 1)
            K = rbf_kernel(Z_i, Z_j, gamma=gamma)
            KB= beta_i.T @ K @ beta_j
            H[i, j] = np.sum(KB)
            if i != j:
                H[j, i] = np.sum(KB)

    if solver=='cvxopt':
        solvers.options['show_progress'] = False
        P=matrix(2*H)    # 0.5 x.T @ P @ x
        q=matrix(-2*C)
        if eq_constraint:
            A=matrix(np.ones((1,c),dtype=float))
            b=matrix([1.0])
        else:
            A=None
            b=None
        if neq_constraint:
            G=matrix(-np.diag(np.ones(c,dtype=float)))
            h=matrix(np.zeros(c,dtype=float))
        else:
            G=None
            h=None

        sol = solvers.qp(P,q,G,h,A,b)
        return np.array(sol['x']).squeeze()
    elif solver=='inv':
        w = np.linalg.pinv(H) @ C
        w = w / np.sum(w)
        return w