import copy
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from core.rkme_generation import Gaussian_Reduced_Set
from core.rkme_match import coefficient_estimation
from core.subspace import SubspaceManager, split_y
from datasets.synthetic_datasets import generate_toy_data
from tools.clf_performance import clf_performance_test
from tools.plot import scatter_with_legends, num2colorTransformer, get_color_list


def main():
    ##################################
    # hyper-parameters configuration #
    ##################################

    random_state = 7

    # paras: learnware jobs generation
    samples_per_cluster = 100
    n_developers = 3

    # paras: RKME
    k = 10
    gamma = 1
    steps = 5
    step_size = 0.1
    constraint_type=1

    # paras: learnware recommendation
    relevance_th = 0.2

    ###################################
    # generate learnware participants #
    ###################################

    # generate developers
    X, X_list, y, indicator = generate_toy_data(random_state, samples_per_cluster)
    C_indices_list = [[0, 1], [1, 2], [0, 2]]

    n = samples_per_cluster
    X_learnware_list = []
    y_learnware_list = []
    for i in range(n_developers):
        C_indices = C_indices_list[i]
        X_temp = np.concatenate(
            [X_list[C_indices[0]][i * 2 * n:(i + 1) *2 * n, :], X_list[C_indices[1]][i *2 * n:(i + 1) *2 * n, :]], axis=1)
        y_temp = y[i * 2 *n:(i + 1) * 2 *n]
        X_learnware_list.append(X_temp)
        y_learnware_list.append(y_temp)

    # generate user
    X_u, X_u_list, y_u, indicator_u = generate_toy_data(random_state, samples_per_cluster)

    X_user_list=[X_u_list[0][2*n:6*n,:],X_u_list[1][2*n:6*n,:]]
    X_user=np.concatenate(X_user_list,axis=1)
    y_user=y_u[2*n:6*n]
    indicator_user=indicator_u[2*n:6*n]

    # generate learnwares
    model_list = []
    for i in range(n_developers):
        clf=SVC()
        # clf = RandomForestClassifier()
        clf.fit(X_learnware_list[i], y_learnware_list[i])
        model_list.append(clf)
        clf_performance_test(model_list[i], X_learnware_list[i], y_learnware_list[i])

    specification_list = []
    for i in range(n_developers):
        rs = Gaussian_Reduced_Set(k=k, gamma=gamma, step_size=step_size, steps=steps, constraint_type=constraint_type)
        rs.fit(X_learnware_list[i])
        rs.beta=np.array(rs.beta)
        specification_list.append(rs)
        print(rs.beta)

    # generate requirement
    requirement = Gaussian_Reduced_Set(k=k, gamma=gamma, step_size=step_size, steps=steps, constraint_type=constraint_type)
    requirement.fit(X_user)
    print(requirement.beta)

    #############################################
    # Construct the dynamic specification space #
    #############################################

    # prepare the input of subspace learning
    Z_c_1 = np.concatenate([specification_list[0].z[:, 0:2], specification_list[2].z[:, 0:2]])
    Z_c_2 = np.concatenate([specification_list[0].z[:, 2:4], specification_list[1].z[:, 0:2]])
    Z_c_3 = np.concatenate([specification_list[1].z[:, 2:4], specification_list[2].z[:, 2:4]])
    indicator_c = [0] * k + [1] * k + [2] * k
    indicator_c=np.array(indicator_c)
    indicator_1 = [0] * k + [2] * k
    indicator_2 = [0] * k + [1] * k
    indicator_3 = [1] * k + [2] * k
    indicator_c_list = [indicator_1, indicator_2, indicator_3]
    Gamma_1 = np.diag(np.concatenate([specification_list[0].beta, specification_list[2].beta]))
    Gamma_2 = np.diag(np.concatenate([specification_list[0].beta, specification_list[1].beta]))
    Gamma_3 = np.diag(np.concatenate([specification_list[1].beta, specification_list[2].beta]))
    Z_c_list = [Z_c_1.T, Z_c_2.T, Z_c_3.T]
    Gamma_list = [Gamma_1, Gamma_2, Gamma_3]

    cardinality_array = [k , k , k ]
    C_indices_list = [[0, 1], [1, 2], [0, 2]]
    R_indices_list = [[0, 2], [0, 1], [1, 2]]

    learner = SubspaceManager(dim=2, alpha=1e-5, beta=1e2, max_iter=1000, learning_rate=5e-2, random_state=random_state,
                              kernel_trick='linear', V_constraint=False)
    # learner = SubspaceManager(dim=2, alpha=1e-3, beta=1e2, max_iter=50, learning_rate=5e-2, random_state=random_state,
    #                           kernel_trick='linear', V_constraint=False)    # final acc: 0.9875
    V_comp, V_list, V_star_list = learner.fit(Z_c_list, Gamma_list, C_indices_list, R_indices_list, cardinality_array)

    Z_pre_list = []
    for i in range(3):
        Z_pre = learner.predict([Z_c_list[i]], [i])
        Z_pre_list.append(Z_pre)

    # market: project the specifications
    Z_pre_list_2 = []
    Z_list_2 = [[Z_c_list[0][:, 0:k], Z_c_list[1][:, 0:k]],
                [Z_c_list[1][:, k:2*k], Z_c_list[2][:, 0:k]],
                [Z_c_list[0][:, k:2*k], Z_c_list[2][:, k:2*k]]]
    for i in range(3):
        Z_pre = learner.predict(Z_list_2[i], C_indices_list[i])
        Z_pre_list_2.append(Z_pre)
    Z_pre = np.concatenate(Z_pre_list_2)

    # market: project the user requirement
    V_user = learner.predict([requirement.z[:,0:2].T,requirement.z[:,2:4].T], [0,1])
    V_user_total = learner.predict([X_user_list[0].T,X_user_list[1].T],[0,1])

    requirement_proj=copy.deepcopy(requirement)
    requirement_proj.z=V_user

    ############################
    # Learnware recommendation #
    ############################

    specification_proj_list = copy.deepcopy(specification_list)
    for i in range(n_developers):
        specification_proj_list[i].z = Z_pre[i * k:(i + 1) * k, :]

    # market: learnware recommendation
    sol = coefficient_estimation(specification_proj_list, V_user, weights=requirement.beta,
                                 solver='cvxopt')
    print('relevance estimation:', sol)
    selected_learnware_idx_list = []
    for i in range(n_developers):
        if sol[i] > relevance_th:
            selected_learnware_idx_list.append(i)
    print('selected learnware idx:', selected_learnware_idx_list)

    ###################
    # Learnware reuse #
    ###################

    X_train_sub_list=[]
    indicator_sub_list=[]
    for idx in selected_learnware_idx_list:
        X_train_sub_list.append(specification_proj_list[idx].z)
        indicator_sub_list.append([idx]*k)
    X_train_sub=np.concatenate(X_train_sub_list)
    indicator_train_sub=np.concatenate(indicator_sub_list)
    clf=SVC()
    clf.fit(X_train_sub, indicator_train_sub)
    indicator_pre=clf.predict(V_user_total)
    print('acc of model selector:',accuracy_score(indicator_user, indicator_pre))

    X_user_total_list=[X_user_list[0],X_user_list[1], learner.transfer(V_user_total,2).T]
    X_user_0=np.concatenate([X_user_list[1],X_user_total_list[2]], axis=1)
    X_user_1 = np.concatenate([X_user_list[0],X_user_total_list[2]], axis=1)
    y_0=model_list[selected_learnware_idx_list[0]].predict(X_user_0)
    y_1=model_list[selected_learnware_idx_list[1]].predict(X_user_1)
    y_list=[0,y_0,y_1]
    final_y=[]
    for i in range(X_user_0.shape[0]):
        final_y.append(y_list[indicator_pre[i]][i])
    final_y=np.array(final_y)
    print('final acc:', accuracy_score(y_user,final_y))

    #################
    # visualization #
    #################

    # basic setup
    figsize=[3,3]
    legend_font_size = 16

    transfomer=num2colorTransformer(get_color_list())

    plot_specifications(xlim=[-1,0.2], ylim=[-1.05,-0.1], specification_list=[specification_list[0],specification_list[2]],
                        dim_list=[0,1,0,1], color_list=[transfomer.transform([0]),transfomer.transform([2])],
                        legend_list=['LW #1','LW #3'], idx_list=[2,3], figsize=figsize, legend_font_size=legend_font_size)
    plt.tight_layout()
    plt.savefig('figure/toy_p1.pdf', dpi=480)

    plt.figure(figsize=figsize)
    plt.xlim([-1, 0.2])
    plt.ylim([-1.05, -0.1])
    plt.xticks([])
    plt.yticks([])
    plt.scatter(X_user_list[0][:,0],X_user_list[0][:,1],facecolor='gray', s=10, alpha=0.5)
    plt.legend(['User (ori.)'],fontsize=legend_font_size, markerscale=2, handletextpad=0)
    plt.tight_layout()
    plt.savefig('figure/toy_p5.pdf', dpi=480)

    plot_specifications(xlim=[-0.1, 1.1], ylim=[-0.9, 1.1],
                        specification_list=[specification_list[0], specification_list[1]],
                        dim_list=[2, 3, 0, 1], color_list=[transfomer.transform([0]), transfomer.transform([1])],
                        legend_list=['LW #1', 'LW #2'], idx_list=[2, 4], figsize=figsize, legend_font_size=legend_font_size)
    plt.tight_layout()
    plt.savefig('figure/toy_p2.pdf', dpi=480)

    plt.figure(figsize=figsize)
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.9, 1.1])
    plt.xticks([])
    plt.yticks([])
    plt.scatter(X_user_list[1][:, 0], X_user_list[1][:, 1], facecolor='gray', s=10, alpha=0.5)
    plt.legend(['User (ori.)'],fontsize=legend_font_size, markerscale=2, handletextpad=0)
    plt.tight_layout()
    plt.savefig('figure/toy_p6.pdf', dpi=480)

    plot_specifications(xlim=[0.1, 1], ylim=[0.2, 1.4],
                        specification_list=[specification_list[1], specification_list[2]],
                        dim_list=[2, 3, 2, 3], color_list=[transfomer.transform([1]), transfomer.transform([2])],
                        legend_list=['LW #2', 'LW #3'], idx_list=[4, 3], figsize=figsize, legend_font_size=legend_font_size)
    plt.tight_layout()
    plt.savefig('figure/toy_p3.pdf', dpi=480)

    plt.figure(figsize=figsize)
    plt.xlim([0.1, 1])
    plt.ylim([0.2, 1.4])
    plt.xticks([])
    plt.yticks([])
    plt.scatter(X_user_total_list[2][:, 0], X_user_total_list[2][:, 1], facecolor='gray', s=10, alpha=0.5, marker='*')
    plt.legend(['User (tran.)'],fontsize=legend_font_size, markerscale=2, handletextpad=0)
    plt.tight_layout()
    plt.savefig('figure/toy_p7.pdf', dpi=480)

    plot_specifications(xlim=[-0.5, 1.25], ylim=[-0.6, 1.2],
                        specification_list=specification_proj_list,
                        dim_list=[0, 1, 0, 1, 0, 1], color_list=[transfomer.transform([0]), transfomer.transform([1]), transfomer.transform([2])],
                        legend_list=['LW #1', 'LW #2', 'LW #3'], idx_list=[2, 4, 3], figsize=figsize, legend_font_size=legend_font_size)
    plt.tight_layout()
    plt.savefig('figure/toy_p4.pdf', dpi=480)

    plot_specifications(xlim=[-0.5, 1.25], ylim=[-0.6, 1.2],
                        specification_list=[requirement_proj],
                        dim_list=[0, 1], color_list=['gray'],
                        legend_list=['Req.'], idx_list=[0], figsize=figsize, legend_font_size=legend_font_size)
    plt.tight_layout()
    plt.savefig('figure/toy_p8.pdf', dpi=480)

    # test
    # plot loss
    plt.figure()
    plt.plot(learner.loss_array)

    plt.show()

def plot_specifications(xlim, ylim, specification_list, dim_list, color_list, legend_list, idx_list, test_mode=False, figsize=[4,4], marker_size=20, legend_font_size=12):
    plt.figure(figsize=figsize)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks([])
    plt.yticks([])
    n=len(specification_list)
    for i in range(n):
        rs = specification_list[i]
        color = color_list[i]
        for j in range(len(rs.z)):
            z = rs.z[j]
            beta = abs(rs.beta[j])
            if not test_mode:
                if j == idx_list[i]:
                    plt.scatter(z[dim_list[2*i]], z[dim_list[2*i+1]], facecolor=color,
                                s=marker_size * 20 * beta, linewidth=0.3, alpha=1, edgecolors=color,
                                marker='o', zorder=2, label=legend_list[i])
                else:
                    plt.scatter(z[dim_list[2*i]], z[dim_list[2*i+1]], facecolor=color,
                                s=marker_size * 20 * beta, linewidth=0.3, alpha=1, edgecolors=color,
                                marker='o', zorder=2)
            else:
                plt.scatter(z[dim_list[2*i]], z[dim_list[2*i+1]], facecolor=color,
                            s=marker_size * 20 * beta, linewidth=0.3, alpha=1, edgecolors=color,
                            marker='o', zorder=2, label=legend_list[i])
    plt.legend(fontsize=legend_font_size, handletextpad=0)

if __name__ == '__main__':
    main()