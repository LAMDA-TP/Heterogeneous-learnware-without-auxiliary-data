from cvxopt import solvers, matrix
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans
from collections import Counter


class Gaussian_Reduced_Set():

    def __init__(self, k, gamma=1, random_seed=0, step_size=0.1, steps=5, constraint_type=1):
        """

        :param k: size of reduced set
        :param gamma: Gaussian kernel parameter
        :param random_seed: Control the randomness of k-means initialization
        :param step_size:
        :param steps:
        :param constraint_type: {0,1,2}
            0: no constraint
            1: non-negative constraint
            2: simplex constraint
        """
        self.z = []
        self.beta = []

        self.k = k
        self.gamma = gamma
        self.random_seed = random_seed
        self.step_size = step_size
        self.steps = steps

        self.constraint_type = constraint_type

    def fit(self, X):
        alpha = np.ones(X.shape[0]) / X.shape[0]
        # Initialize Z by K-means
        self.init_z_by_kmeans(X, self.k)
        self.update_beta(alpha, X)
        # Alternating optimize Z and beta
        for i in range(self.steps):
            self.update_z(alpha, X, self.step_size)
            self.update_beta(alpha, X)

        self.z = np.array(self.z)

    def init_z_by_kmeans(self, X, k):
        kmeans = KMeans(n_clusters=k, random_state=self.random_seed, n_init=10)
        kmeans.fit(X)
        self.z = list(kmeans.cluster_centers_)

    def update_beta(self, alpha, X):
        Z = np.array(self.z)
        K_z = rbf_kernel(Z, gamma=self.gamma)
        K_zx = rbf_kernel(Z, X, gamma=self.gamma)

        if self.constraint_type == 0:  # no constraint
            beta = np.linalg.pinv(K_z) @ K_zx @ alpha
            self.beta = list(beta)
        else:
            solvers.options['show_progress'] = False
            P = matrix(2 * K_z)
            q = matrix(-2 * K_zx @ alpha)
            if self.constraint_type == 1:  # non-negative constraint
                A = None
                b = None
                G = matrix(-np.eye(self.k, dtype=float))
                h = matrix(np.zeros(self.k, dtype=float))
            elif self.constraint_type == 2:  # simplex constraint
                A = matrix(np.ones((1, self.k), dtype=float))
                b = matrix([1.0])
                G = matrix(-np.eye(self.k, dtype=float))
                h = matrix(np.zeros(self.k, dtype=float))
            else:
                raise Exception('invalid constraint type')
            sol = solvers.qp(P, q, G, h, A, b)
            self.beta = list(np.array(sol['x']).squeeze())

    def update_z(self, alpha, X, step_size):
        gamma = self.gamma
        Z = np.array(self.z)
        grad_Z = np.zeros(Z.shape)
        beta = np.array(self.beta)
        for i in range(Z.shape[0]):
            z_i = Z[i, :].reshape(1, -1)
            term_1 = 2 * (beta * rbf_kernel(z_i, Z, gamma)) @ (z_i - Z)
            term_2 = -2 * (alpha * rbf_kernel(z_i, X, gamma)) @ (z_i - X)
            grad_Z[i, :] = -2 * gamma * beta[i] * (term_1 + term_2)
        Z = Z - step_size * grad_Z
        self.z = list(Z)

    def sampling_candidates(self, N):
        np.random.seed(self.random_seed)
        m = len(self.beta)
        d = self.z[0].shape[0]
        beta = np.array(self.beta)
        beta[beta < 0] = 0  # currently we cannot use negative weight
        beta = beta / np.sum(beta)
        sample_assign = np.random.choice(m, size=N, p=beta)
        sample_list = []
        for i, n in Counter(sample_assign).items():
            sample_list.append(np.random.normal(loc=self.z[i], scale=self.gamma, size=(n, d)))
        if len(sample_list) > 1:
            return np.concatenate(sample_list, axis=0)
        elif len(sample_list) == 1:
            return sample_list[0]
        else:
            print("error")

    def herding(self, N, super_sampling_ratio=100):
        # Generate a large number of candidates for herding
        Nstart = super_sampling_ratio * N
        Xstart = self.sampling_candidates(Nstart)
        if Nstart >= 1000:
            return Xstart
        else:
            D = self.z[0].shape[0]
            S = np.zeros((N, D))
            fsX = self.eval(Xstart)
            fsS = np.zeros(Nstart)
            for i in range(N):
                if i > 0:
                    fsS = np.sum(rbf_kernel(S[:i, :], Xstart, self.gamma), axis=0)
                fs = (i + 1) * fsX - fsS
                idx = np.argmax(fs)
                S[i, :] = Xstart[idx, :]
        return S

    def eval(self, X):
        # Compute Phi(X)
        Z = np.array(self.z)
        beta = np.array(self.beta)
        v = np.zeros((Z.shape[0], X.shape[0]))
        for i in range(Z.shape[0]):
            z_i = self.z[i].reshape(1, -1)
            v[i, :] = beta[i] * rbf_kernel(z_i, X, gamma=self.gamma)
        v = np.sum(v, axis=0)
        return v

    def approx_error(self, X, alpha):
        beta = np.array(self.beta)
        z = np.array(self.z)
        x1 = rbf_kernel(X, X, gamma=self.gamma)
        x2 = rbf_kernel(z, z, gamma=self.gamma)
        x3 = rbf_kernel(X, z, gamma=self.gamma)
        x1 = alpha.T @ x1 @ alpha
        x2 = beta.T @ x2 @ beta
        x3 = alpha.T @ x3 @ beta
        return x1 + x2 - 2 * x3
