import numpy as np

from core.subspace_base import SubspaceBase
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, cosine_similarity


class SubspaceManager(SubspaceBase):

    def __init__(self, dim, alpha=1.0, beta=1.0, kernel_trick='linear', kr_gamma=None, max_iter=100, learning_rate=1e-3,
                 sim_kernel='rbf', n_neighbors=5, sim_gamma=None, random_state=None, V_constraint=False):
        super().__init__(random_state=random_state)
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.kernel_trick = kernel_trick
        self.kr_gamma = kr_gamma
        self.sim_kernel = sim_kernel
        self.n_neighbors = n_neighbors
        self.sim_gamma = sim_gamma
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.V_constraint = V_constraint

        np.random.seed(random_state)

    def obj_base(self, K, W, V, Gamma, L, V_star):
        """

        Args:
            K: n*n
            W: n*k
            V: n*k
            Gamma: n*n
            L: n*n
            V_star: n*k

        Returns:

        """
        if Gamma is None:
            term_11 = np.trace(K)
            term_12 = np.trace(K @ W @ V.T)
            term_13 = np.trace(V @ W.T @ K @ W @ V.T)
        else:
            term_11 = np.trace(Gamma @ K)
            term_12 = np.trace(Gamma @ K @ W @ V.T)
            term_13 = np.trace(Gamma @ V @ W.T @ K @ W @ V.T)
        term_2 = np.trace(V.T @ L @ V)
        if Gamma is None:
            term_3 = np.linalg.norm(V - V_star, ord='fro') ** 2
        else:
            term_3=np.trace(V.T @ Gamma @ V) -2 * np.trace(V_star.T @ Gamma @ V) + np.trace(V_star.T @ Gamma @ V_star)
        return term_11 - 2 * term_12 + term_13 + self.alpha * term_2 + self.beta * term_3

    def obj(self, K_list, W_list, V_list, Gamma_list, L_list, V_star_list):
        m = len(K_list)
        obj_temp = 0
        for i in range(m):
            obj_temp += self.obj_base(K_list[i], W_list[i], V_list[i], Gamma_list[i], L_list[i], V_star_list[i])
        return obj_temp

    def update_W(self, K, W, V, K_pos, K_neg, Gamma=None):
        if Gamma is None:
            temp_pos, temp_neg = self.pos_neg_split(V.T @ V)
            P = K_pos @ W @ temp_pos + K_neg @ W @ temp_neg
            N = K_pos @ W @ temp_neg + K_neg @ W @ temp_pos
            return self.MUR(W, K @ V, P, N)
        else:
            temp_pos, temp_neg = self.pos_neg_split(V.T @ Gamma @ V)
            P = K_pos @ W @ temp_pos + K_neg @ W @ temp_neg
            N = K_pos @ W @ temp_neg + K_neg @ W @ temp_pos
            return self.MUR(W, K @ Gamma @ V, P, N)

    def update_V(self, K, W, V, Gamma, L, V_star, D, S):
        if not self.V_constraint:
            if Gamma is None:
                gradient = -2 * K @ W + 2 * V @ W.T @ K @ W + 2 * self.alpha * L @ V + 2 * self.beta * V - 2 * self.beta * V_star
            else:
                gradient = -2 * Gamma @ K @ W + 2 * Gamma @ V @ W.T @ K @ W + 2 * self.alpha * L @ V + 2 * self.beta * Gamma @ V - 2 * self.beta * Gamma @ V_star
            return V - self.learning_rate * gradient
        else:
            if Gamma is None:
                K_pos,K_neg=self.pos_neg_split(K)
                P = V @ W.T @ K_pos @ W + self.alpha * D @ V + self.beta * V
                N = V @ W.T @ K_neg @ W + self.alpha * S @ V
                return self.MUR(V, K @ W + self.beta * V_star, P, N)
            else:
                Gamma_pos, Gamma_neg=self.pos_neg_split(Gamma)
                K_pos, K_neg = self.pos_neg_split(K)
                P=Gamma_pos @ V @ W.T @ K_pos @ W + Gamma_neg @ V @ W.T @ K_neg @ W + self.alpha * D @ V + self.beta * Gamma_pos @ V
                N=Gamma_pos @ V @ W.T @ K_neg @ W + Gamma_neg @ V @ W.T @ K_pos @ W + self.alpha * S @ V + self.beta * Gamma_neg @ V
                return self.MUR(V, Gamma @ K @ W + self.beta * Gamma @ V_star, P, N)

    def update_V_star(self, V_list, C_indices_list, R_indices_list, cardinality_array):
        V_list_block = self.split_V_list(V_list, R_indices_list, cardinality_array)

        T = len(C_indices_list)
        V_comp_list = []
        for i in range(T):
            V_comp = 0
            for k in C_indices_list[i]:
                V_comp += V_list_block[k][i]
            V_comp /= len(C_indices_list[i])
            V_comp_list.append(V_comp)
        V_star_list = []
        m = len(V_list_block)
        for k in range(m):
            V_star_temp_list = [V_comp_list[i] for i in R_indices_list[k]]
            V_star = np.concatenate(V_star_temp_list)
            V_star_list.append(V_star)

        V_comp=np.concatenate(V_comp_list)

        return V_star_list, V_comp

    def split_V(self, V, R_indices, cardinality_array):
        n_total_block = len(cardinality_array)
        V_block = []
        start = 0
        end = 0
        for i in range(n_total_block):
            if i in R_indices:
                end += cardinality_array[i]
                V_block.append(V[start:end, :])
                start += cardinality_array[i]
            else:
                V_block.append([])
        return V_block

    def split_V_list(self, V_list, R_indices_list, cardinality_array):
        m = len(V_list)
        V_block_list = []
        for i in range(m):
            V_block = self.split_V(V_list[i], R_indices_list[i], cardinality_array)
            V_block_list.append(V_block)
        return V_block_list

    def initialization_base(self, Z, K, sample_weights=None):
        W, V = self.W_init(Z, self.dim, 'kmeans', sample_weights)
        # if not self.V_constraint:
        #     V_T = np.linalg.pinv(W.T @ K @ W) @ W.T @ K
        #     V = V_T.T
        W, V = self.normalization(W, V, normalization_type='MAX')
        return W, V

    def initialization(self, Z_list, K_list, Gamma_list, C_indices_list, R_indices_list, cardinality_array):
        m = len(Z_list)
        W_list, V_list = [], []
        for k in range(m):
            if Gamma_list is None or Gamma_list[k] is None:
                sample_weights = None
            else:
                sample_weights = np.diagonal(Gamma_list[k])
            W, V = self.initialization_base(Z_list[k], K_list[k], sample_weights)
            W_list.append(W)
            V_list.append(V)
        V_star_list, V_comp = self.update_V_star(V_list, C_indices_list, R_indices_list, cardinality_array)
        return W_list, V_list, V_star_list, V_comp

    def fit(self, Z_list, Gamma_list, C_indices_list, R_indices_list, cardinality_array):
        T = len(C_indices_list)
        m = len(R_indices_list)

        K_list, K_pos_list, K_neg_list = [], [], []
        L_list, D_list, S_list = [], [], []
        for i in range(m):
            K = self.generate_kernel_matrix(Z_list[i], kernel_trick=self.kernel_trick, gamma=self.kr_gamma)
            K_pos, K_neg = self.pos_neg_split(K)
            K_list.append(K)
            K_pos_list.append(K_pos)
            K_neg_list.append(K_neg)

            L, D, S = self.generate_laplacian_matrix(Z_list[i], sim_kernel=self.sim_kernel,
                                                     n_neighbors=self.n_neighbors,
                                                     gamma=self.sim_gamma)
            L_list.append(L)
            D_list.append(D)
            S_list.append(S)

        W_list, V_list, V_star_list, V_comp = self.initialization(Z_list, K_list, Gamma_list, C_indices_list, R_indices_list,
                                                          cardinality_array)

        loss_array=[]
        for cur_iter in range(self.max_iter):
            for k in range(m):
                W_list[k] = self.update_W(K_list[k], W_list[k], V_list[k], K_pos_list[k], K_neg_list[k], Gamma_list[k])
                V_list[k] = self.update_V(K_list[k], W_list[k], V_list[k], Gamma_list[k], L_list[k], V_star_list[k], D_list[k], S_list[k])
                W_list[k], V_list[k] = self.normalization(W_list[k], V_list[k], normalization_type='MAX')
            V_star_list, V_comp = self.update_V_star(V_list, C_indices_list, R_indices_list, cardinality_array)
            obj_val = self.obj(K_list, W_list, V_list, Gamma_list, L_list, V_star_list)
            print(cur_iter, obj_val)

            loss_array.append(obj_val)

        self.loss_array=loss_array

        self.Z_list=Z_list
        self.W_list=W_list

        return V_comp, V_list, V_star_list

    def predict_base(self, X_u, col_index):
        """

        Args:
            X_u: d*n

        Returns:

        """
        if self.kernel_trick=='linear':
            B=self.Z_list[col_index] @ self.W_list[col_index]
            X_pre = np.linalg.pinv(B.T @ B) @ B.T @ X_u
            X_pre = X_pre.T
        elif self.kernel_trick=='rbf':
            K = rbf_kernel(self.Z_list[col_index].T)
            K_u=rbf_kernel(self.Z_list[col_index].T, X_u.T)
            W=self.W_list[col_index]
            X_pre = np.linalg.pinv(W.T @ K @ W) @ W.T @ K_u
            X_pre = X_pre.T

        return X_pre

    def predict(self, X_u, C_indices):
        """

        Args:
            X_u: [d_1*n,\cdots, d_k*n]
            C_indices:

        Returns:
            n*d_sub
        """
        X_pre=0
        for i,col_index in enumerate(C_indices):
            X_pre+=self.predict_base(X_u[i], col_index)
        X_pre=X_pre/len(C_indices)
        return X_pre

    def transfer(self, V_u, idx):
        if self.kernel_trick=='linear':
            return self.Z_list[idx] @ self.W_list[idx] @ V_u.T

def split_y(y, R_indices,cardinality_array):
    T=len(cardinality_array)
    new_y=[]
    start = 0
    end = 0
    for i in range(T):
        end += cardinality_array[i]
        if i in R_indices:
            new_y.append(y[start:end])
        start += cardinality_array[i]
    new_y = np.concatenate(new_y)
    return new_y
