import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, cosine_similarity
from sklearn.neighbors import kneighbors_graph


class SubspaceBase(object):

    def __init__(self, random_state=None):
        self.random_state = random_state

    def MUR(self, X, B, P, N, small_val_th=1e-9):
        assert (P >= 0).all()
        assert (N >= 0).all()

        P[P < small_val_th] = small_val_th  # avoid the denominator is 0

        temp = B * B + 4 * P * N
        temp = np.sqrt(temp)  # sqrt may need similar pre-processing
        temp += B
        return X * temp / (2 * P)

    def pos_neg_split(self, X):
        X_pos = X.copy()
        X_pos[X_pos < 0] = 0
        X_neg = -X.copy()
        X_neg[X_neg < 0] = 0
        return X_pos, X_neg

    def W_init(self, X, dim_subspace, clustering_method='kmeans', sample_weights=None):
        """

        Args:
            X: d*n
            dim_subspace:
            clustering_method: {'rand', 'kmeans'}
            sample_weights:

        Returns:

        """
        n = X.shape[1]
        if clustering_method == 'rand':
            W = 1 - np.random.rand(n, dim_subspace)  # rand: [0,1)  1-rand: (0,1]
            V = W.copy()
        elif clustering_method == 'kmeans':
            alg = KMeans(n_clusters=dim_subspace, random_state=self.random_state, n_init=10)

            y = alg.fit_predict(X.T, sample_weight=sample_weights)
            n = X.shape[1]
            V = np.ones(shape=(n, dim_subspace)) * 0.1
            cluster_size = np.zeros(dim_subspace)
            for i in range(n):
                V[i][y[i]] += 1
                cluster_size[y[i]] += 1
            W = V @ np.diag(1 / cluster_size)

        return W, V

    def normalization(self, W, V, normalization_type='MAX'):
        """

        Args:
            W: n*k
            V: n*k
            normalization_type: {'L1','L2','MAX'}

        Returns:

        """
        if normalization_type == 'L1':
            diag_elements = np.linalg.norm(V, ord=1, axis=0, keepdims=False)
        elif normalization_type == 'L2':
            diag_elements = np.sqrt(np.linalg.norm(V, ord=2, axis=0, keepdims=False))
        elif normalization_type == 'MAX':
            diag_elements = np.linalg.norm(V, ord=np.inf, axis=0, keepdims=False)

        diag_elements_inv = np.array([1 / x if x > 0 else 0 for x in diag_elements])
        Lambda = np.diag(diag_elements)
        Lambda_inv = np.diag(diag_elements_inv)
        new_V = V @ Lambda_inv
        new_W = W @ Lambda
        return new_W, new_V

    def generate_kernel_matrix(self, X, kernel_trick='linear', gamma=None):
        """

        Args:
            X: d*n
            kernel_trick: {'linear', 'rbf'}
            gamma: paras for rbf kernel

        Returns:
            K: n*n
        """
        if kernel_trick == 'linear':
            K = linear_kernel(X.T)
        elif kernel_trick == 'rbf':
            K = rbf_kernel(X.T, gamma=gamma)
        return K

    def generate_laplacian_matrix(self, X, sim_kernel='rbf', n_neighbors=5, gamma=None):
        """

        Args:
            X: d*n
            sim_kernel: {'linear','cos','rbf'}
            n_neighbors:
            gamma:

        Returns:

        """
        if sim_kernel == 'linear':
            sim_mat = linear_kernel(X.T)
        elif sim_kernel == 'cos':
            sim_mat = cosine_similarity(X.T)
        elif sim_kernel == 'rbf':
            sim_mat = rbf_kernel(X.T, gamma=gamma)
        neighbor_graph = kneighbors_graph(X.T, n_neighbors=n_neighbors)
        S = sim_mat * neighbor_graph
        D = np.diag(np.sum(S, axis=0))
        L = D - S
        return L, D, S


class SubspaceLearner(SubspaceBase):

    def __init__(self, dim, alpha=1e-2, kernel_trick='linear', kr_gamma=None, sim_kernel='rbf', n_neighbors=5, sim_gamma=None,
                 max_iter=100, learning_rate=1e-3, init='kmeans', normalization_strategy='MAX', random_state=None,
                 V_constraint=False):
        """

        Args:
            dim:
            alpha:
            learning_rate:
            init:
            normalization_strategy: normalization method only for training,
                the final normalization for the outcomes is 'MAX'
            random_state:
        """
        super().__init__(random_state=random_state)
        self.dim = dim
        self.alpha = alpha
        self.kernel_trick = kernel_trick
        self.kr_gamma = kr_gamma
        self.sim_kernel = sim_kernel
        self.n_neighbors = n_neighbors
        self.sim_gamma = sim_gamma
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.init = init
        self.normalization_strategy = normalization_strategy
        self.random_state = random_state
        self.V_constraint=V_constraint

        np.random.seed(random_state)

    def obj(self, K, W, V, L, Gamma=None):
        if Gamma is None:
            term_1 = np.trace(K)
            term_2 = np.trace(K @ W @ V.T)
            term_3 = np.trace(V @ W.T @ K @ W @ V.T)
            term_4 = np.trace(V.T @ L @ V)
        else:
            term_1 = np.trace(Gamma @ K)
            term_2 = np.trace(Gamma @ K @ W @ V.T)
            term_3 = np.trace(Gamma @ V @ W.T @ K @ W @ V.T)
            term_4 = np.trace(V.T @ L @ V)
        return term_1 - 2 * term_2 + term_3 + self.alpha * term_4

    def initialization(self, X, K, sample_weights=None):
        W, V = self.W_init(X, self.dim, self.init, sample_weights)
        if not self.V_constraint:
            V_T = np.linalg.pinv(W.T @ K @ W) @ W.T @ K
            V = V_T.T
        W, V = self.normalization(W, V, normalization_type=self.normalization_strategy)
        return W, V

    def update_W(self, K, W, V, K_pos, K_neg, Gamma=None):
        if not self.V_constraint:
            if Gamma is None:
                temp_pos,temp_neg=self.pos_neg_split(V.T @ V)
                P = K_pos @ W @ temp_pos + K_neg @ W @ temp_neg
                N = K_pos @ W @ temp_neg + K_neg @ W @ temp_pos
                return self.MUR(W, K @ V, P, N)
            else:
                temp_pos, temp_neg = self.pos_neg_split(V.T @ Gamma @ V)
                P = K_pos @ W @ temp_pos + K_neg @ W @ temp_neg
                N = K_pos @ W @ temp_neg + K_neg @ W @ temp_pos
                return self.MUR(W, K @ Gamma @ V, P, N)
        else:
            if Gamma is None:
                P = K_pos @ W @ V.T @ V
                N = K_neg @ W @ V.T @ V
                return self.MUR(W, K @ V, P, N)
            else:
                temp_pos, temp_neg = self.pos_neg_split(Gamma)
                P = K_pos @ W @ V.T @ temp_pos @ V + K_neg @ W @ V.T @ temp_neg @ V
                N = K_neg @ W @ V.T @ temp_pos @ V + K_pos @ W @ V.T @ temp_neg @ V
                return self.MUR(W, K @ Gamma @ V, P, N)

    def update_V(self, K, W, V, L, Gamma, D, S):
        if not self.V_constraint:
            if Gamma is None:
                gradient = -2 * K @ W + 2 * V @ W.T @ K @ W + 2 * self.alpha * L @ V
            else:
                gradient = -2 * Gamma @ K @ W + 2 * Gamma @ V @ W.T @ K @ W + 2 * self.alpha * L @ V
            return V - self.learning_rate * gradient
        else:
            if Gamma is None:
                K_pos,K_neg=self.pos_neg_split(K)
                P = V @ W.T @ K_pos @ W + self.alpha * D @ V
                N = V @ W.T @ K_neg @ W + self.alpha * S @ V
                return self.MUR(V, K @ W, P, N)
            else:
                Gamma_pos, Gamma_neg=self.pos_neg_split(Gamma)
                K_pos, K_neg = self.pos_neg_split(K)
                P=Gamma_pos @ V @ W.T @ K_pos @ W + Gamma_neg @ V @ W.T @ K_neg @ W + self.alpha * D @ V
                N=Gamma_pos @ V @ W.T @ K_neg @ W + Gamma_neg @ V @ W.T @ K_pos @ W+ self.alpha * S @ V
                return self.MUR(V, Gamma @ K @ W, P, N)

    def fit(self, X, sample_weights=None):
        """

        Args:
            X: d*n
            sample_weights:

        Returns:

        """
        K = self.generate_kernel_matrix(X, kernel_trick=self.kernel_trick, gamma=self.kr_gamma)
        K_pos, K_neg=self.pos_neg_split(K)
        L, D, S = self.generate_laplacian_matrix(X, sim_kernel=self.sim_kernel, n_neighbors=self.n_neighbors,
                                                 gamma=self.sim_gamma)
        W, V = self.initialization(X, K, sample_weights)
        if sample_weights is None:
            Gamma=None
        else:
            Gamma=np.diag(sample_weights)
        for cur_iter in range(self.max_iter):
            W = self.update_W(K, W, V, K_pos, K_neg, Gamma)
            V = self.update_V(K, W, V, L, Gamma, D, S)
            W, V = self.normalization(W, V, normalization_type=self.normalization_strategy)
            obj_val = self.obj(K, W, V, L, Gamma)
            print(cur_iter, obj_val)
        W, V = self.normalization(W, V, normalization_type='MAX')

        self.X=X
        self.W=W

        return W, V

    def predict(self, X_u):
        """

        Args:
            X_u: d*n

        Returns:

        """
        if self.kernel_trick=='linear':
            B=self.X @ self.W
            X_pre = np.linalg.pinv(B.T @ B) @ B.T @ X_u
            X_pre = X_pre.T
        elif self.kernel_trick=='rbf':
            K = rbf_kernel(self.X.T)
            K_u=rbf_kernel(self.X.T, X_u.T)
            X_pre = np.linalg.pinv(W.T @ K @ W) @ W.T @ K_u
            X_pre = X_pre.T

        return X_pre

    def reconstruct(self, V_u):
        """

        Args:
            V_u: k*n

        Returns:
            reconstructed matrix d*n
        """
        if self.kernel_trick=='linear':
            return self.X @ self.W @ V_u
        else:
            print("reconstruction is invalid for kernel trick.")
