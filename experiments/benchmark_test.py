"""
This file conducts the learnware experiments on the benchmarks and the real-world project
"""
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from core.rkme_generation import Gaussian_Reduced_Set
from core.rkme_match import coefficient_estimation, mmd_distance_list
from core.subspace import SubspaceManager
from datasets.benchmarks import DatasetLoader, describe_dataset
from datasets.learnware_task_generation import LearnwareScenarioGenerator, describe_learnware_scenario_market, describe_learnware_scenario_user
from experiments.paras_config import paras_setup, experiment_paras, experiment_supp_paras

def run_learnware_procedure(task_name, list_generation_type='split',
                            developer_model=SVC(kernel='linear', probability=True),
                            rkme_size = 10, rkme_gamma = 1e-2, rkme_steps = 5, rkme_step_size = 0.1, rkme_constraint_type = 1,
                            sl_dim=10, sl_alpha=1e-5, sl_beta=1e0, sl_max_iter=500, sl_learning_rate=1e-2,
                            C_indices_user=[1,2],
                            relevance_th = 0.1, n_repeated=50, verbose=1):

    ##################################
    # extra-parameters configuration #
    ##################################
    global_random_state=0

    class_assignment_list=paras_setup[task_name]['class_assignment_list']
    C_indices_list=paras_setup[task_name]['C_indices_list']
    R_indices_list = paras_setup[task_name]['R_indices_list']

    ###########################################
    # Load dataset according to the task name #
    ###########################################
    if verbose>=1:
        print('Load dataset according to the task name...')
    dataset_loader=DatasetLoader(data_home='../datasets/benchmarks')
    X_list, y=dataset_loader.load_task_dataset(task_name, list_generation_type=list_generation_type)
    dim_list=get_dim_list(X_list)
    if verbose>=1:
        describe_dataset(X_list, y, task_name)

    ###############################
    # Generate learnware scenario #
    ###############################
    if verbose>=1:
        print('Generate learnware scenario...')
    generator=LearnwareScenarioGenerator(X_list, y, class_assignment_list, global_random_state)
    X_list_dev, y_list_dev=generator.generate_developers(C_indices_list)

    if verbose >= 1:
        clf = deepcopy(developer_model)
        describe_learnware_scenario_market(X_list_dev, y_list_dev, clf)

    #######################################################
    # Developers prepare the model with raw specification #
    #######################################################
    # generate models
    n_developer=len(X_list_dev)
    if verbose>=1:
        print('Generate raw learnwares...')
    model_list=[]
    for i in range(n_developer):
        clf=deepcopy(developer_model)
        clf.fit(X_list_dev[i], y_list_dev[i])
        model_list.append(clf)

    # generate raw specifications
    specification_list = []
    for i in range(n_developer):
        rs = Gaussian_Reduced_Set(k=rkme_size, gamma=rkme_gamma, random_seed=global_random_state, step_size=rkme_step_size, steps=rkme_steps, constraint_type=rkme_constraint_type)
        rs.fit(X_list_dev[i])
        rs.beta=np.array(rs.beta)
        specification_list.append(rs)

    # generate mimic data
    X_mimic_list_all = []
    y_mimic_list_all = []
    for i in range(n_developer):
        if verbose >= 1:
            print('generate the mimic data of %d-th learnware' % i)
        X_mimic = specification_list[i].herding(X_list_dev[i].shape[0], super_sampling_ratio=5)
        y_mimic = model_list[i].predict(X_mimic)
        X_mimic_list_all.append(X_mimic)
        y_mimic_list_all.append(y_mimic)

    ############################
    # Specification adjustment #
    ############################
    if verbose>=1:
        print('Specification and requirement adjustment...')

    Z_list, Gamma_list, cardinality_array, specification_block_list=generate_subspace_learning_input(specification_list, C_indices_list, R_indices_list, dim_list)
    manager = SubspaceManager(dim=sl_dim, alpha=sl_alpha, beta=sl_beta, max_iter=sl_max_iter, learning_rate=sl_learning_rate, random_state=global_random_state,
                              kernel_trick='linear', V_constraint=False)
    manager.fit(Z_list, Gamma_list, C_indices_list, R_indices_list, cardinality_array)

    Z_pre_list=[]
    for i in range(n_developer):
        X_list_temp=specification_block_list[i]
        X_list_temp=[X.T for X in X_list_temp]
        Z_pre=manager.predict(X_list_temp, C_indices_list[i])
        Z_pre_list.append(Z_pre)

    specification_proj_list = deepcopy(specification_list)
    for i in range(n_developer):
        specification_proj_list[i].z=Z_pre_list[i]

    ######################################

    n_mixed_list=[1,2,3]

    random_selection_results=[]
    ensemble_results=[]
    mmd_min_results=[]
    mlj_results=[]
    our_proj_results=[]
    our_trans_results=[]

    with tqdm(total=n_repeated * len(n_mixed_list)) as pbar:
        for n_mixed in n_mixed_list:
            acc_list_random_selection=[]
            acc_list_ensemble=[]
            acc_list_mmd_min=[]
            acc_list_mlj=[]
            acc_list_our_proj=[]
            acc_list_our_trans=[]
            for random_seed_user in range(n_repeated):
                #############################################
                # Generate the user and its raw requirement #
                #############################################

                # C_indices_user=np.random.choice(3,n_blocks, replace=False)
                np.random.seed(random_seed_user)
                selected_task_indices = np.random.choice(6,n_mixed, replace=False)
                print(selected_task_indices)

                X_user, y_user = generator.generate_user(C_indices=C_indices_user, selected_task_indices=selected_task_indices)
                # if verbose >= 1:
                #     clf = SVC()
                #     describe_learnware_scenario_user(X_user, y_user, clf)

                requirement = Gaussian_Reduced_Set(k=rkme_size, gamma=rkme_gamma, random_seed=global_random_state,
                                                   step_size=rkme_step_size, steps=rkme_steps, constraint_type=rkme_constraint_type)
                requirement.fit(X_user)

                # compared methods without specification
                acc_random_selection=cm_random_selection(model_list, C_indices_list, X_user, y_user, C_indices_user, random_seed_user)
                acc_ensemble=cm_ensemble_all(model_list, C_indices_list, X_user, y_user, C_indices_user)

                # compared methods raw specification
                acc_mmd_min=cm_mmd_minimal(model_list, specification_list, C_indices_list, X_user, y_user, C_indices_user)
                acc_mlj=cm_simplified_previous_work(model_list, specification_list, C_indices_list, X_user, y_user, C_indices_user)

                ##########################
                # Requirement adjustment #
                ##########################
                Z_user_list=split_X(requirement.z, dim_list, C_indices_user)
                Z_user_list=[Z.T for Z in Z_user_list]
                Z_user_pre=manager.predict(Z_user_list, C_indices_user)

                requirement_proj = deepcopy(requirement)
                requirement_proj.z = Z_user_pre

                ############################
                # Learnware recommendation #
                ############################
                if verbose>=1:
                    print('Recommend learnwares...')
                # market: learnware recommendation
                sol = coefficient_estimation(specification_proj_list, Z_user_pre, weights=requirement.beta,
                                             solver='cvxopt')
                print('relevance estimation:', sol)
                selected_learnware_idx_list = []
                for i in range(n_developer):
                    if sol[i] > relevance_th:
                        selected_learnware_idx_list.append(i)
                print('selected learnware idx:', selected_learnware_idx_list)

                ###################
                # Learnware reuse #
                ###################

                X_user_list = split_X(X_user, dim_list, C_indices_user)
                V_user = manager.predict([X.T for X in X_user_list], C_indices_user)

                acc_proj=our_projection(manager, X_mimic_list_all, y_mimic_list_all, selected_learnware_idx_list, dim_list, C_indices_list,
                           V_user, y_user)
                acc_trans=our_transfer(manager, model_list, specification_proj_list, C_indices_list, dim_list, rkme_size,
                         V_user, X_user_list, y_user, C_indices_user,
                         selected_learnware_idx_list)

                acc_list_random_selection.append(acc_random_selection)
                acc_list_ensemble.append(acc_ensemble)
                acc_list_mmd_min.append(acc_mmd_min)
                acc_list_mlj.append(acc_mlj)
                acc_list_our_proj.append(acc_proj)
                acc_list_our_trans.append(acc_trans)

                pbar.update(1)
            random_selection_results.append([np.mean(acc_list_random_selection), np.std(acc_list_random_selection)])
            ensemble_results.append([np.mean(acc_list_ensemble), np.std(acc_list_ensemble)])
            mmd_min_results.append([np.mean(acc_list_mmd_min), np.std(acc_list_mmd_min)])
            mlj_results.append([np.mean(acc_list_mlj), np.std(acc_list_mlj)])
            our_proj_results.append([np.mean(acc_list_our_proj), np.std(acc_list_our_proj)])
            our_trans_results.append([np.mean(acc_list_our_trans), np.std(acc_list_our_trans)])
            print('random selection acc: %.3f (%.3f)'%(np.mean(acc_list_random_selection), np.std(acc_list_random_selection)))
            print('ensemble acc: %.3f (%.3f)' % (np.mean(acc_list_ensemble), np.std(acc_list_ensemble)))
            print('mmd min acc: %.3f (%.3f)' % (np.mean(acc_list_mmd_min), np.std(acc_list_mmd_min)))
            print('mlj acc: %.3f (%.3f)' % (np.mean(acc_list_mlj), np.std(acc_list_mlj)))
            print('our projection acc: %.3f (%.3f)' % (np.mean(acc_list_our_proj), np.std(acc_list_our_proj)))
            print('our trans acc: %.3f (%.3f)' % (np.mean(acc_list_our_trans), np.std(acc_list_our_trans)))

    output_info=dict()
    output_info['sl_loss']=manager.loss_array
    output_info['final_results']=[
        random_selection_results, ensemble_results, mmd_min_results, mlj_results, our_proj_results, our_trans_results
    ]

    return output_info


def our_transfer(manager, model_list, specification_proj_list, C_indices_list, dim_list, rkme_size,
                 V_user, X_user_list, y_user, C_indices_user,
                 selected_learnware_idx_list):
    X_user_list_whole = []
    for i in range(len(dim_list)):
        if i in C_indices_user:
            idx_temp = np.argwhere(np.array(C_indices_user) == i).squeeze()
            X_user_list_whole.append(X_user_list[idx_temp])
        else:
            X_transfer = manager.transfer(V_user, i)
            X_user_list_whole.append(X_transfer.T)

    X_train_sub_list = []
    indicator_sub_list = []
    for i, idx in enumerate(selected_learnware_idx_list):
        X_train_sub_list.append(specification_proj_list[idx].z)
        indicator_sub_list.append([i] * rkme_size)
    X_train_sub = np.concatenate(X_train_sub_list)
    indicator_train_sub = np.concatenate(indicator_sub_list)
    if len(set(indicator_train_sub)) == 1:
        indicator_pre = np.array(list(set(indicator_train_sub)) * V_user.shape[0])
    else:
        clf = SVC()
        clf.fit(X_train_sub, indicator_train_sub)
        indicator_pre = clf.predict(V_user)

    prediction_list = []
    for model_idx in selected_learnware_idx_list:
        X_temp_list = []
        for j in C_indices_list[model_idx]:
            X_temp_list.append(X_user_list_whole[j])
        X_temp = np.concatenate(X_temp_list, axis=1)
        y_temp = model_list[model_idx].predict(X_temp)
        prediction_list.append(y_temp)

    final_prediction = []
    for i in range(len(y_user)):
        final_prediction.append(prediction_list[indicator_pre[i]][i])
    final_prediction = np.array(final_prediction)
    acc = accuracy_score(y_user, final_prediction)
    return acc

def our_projection(manager, X_mimic_list_all, y_mimic_list_all, selected_learnware_idx_list, dim_list, C_indices_list,
                   V_user, y_user):
    X_mimic_list = []
    y_mimic_list = []
    for idx in selected_learnware_idx_list:
        X_mimic_list.append(X_mimic_list_all[idx])
        y_mimic_list.append(y_mimic_list_all[idx])

    # user: project the mimic data
    X_mimic_proj_list = []
    for i, idx in enumerate(selected_learnware_idx_list):
        X_list_temp = split_X(X_mimic_list[i], dim_list, C_indices_list[idx])
        X_list_temp = [X.T for X in X_list_temp]
        projection = manager.predict(X_list_temp, C_indices_list[idx])
        X_mimic_proj_list.append(projection)

    # user: make a prediction
    clf = SVC()
    clf.fit(np.concatenate(X_mimic_proj_list), np.concatenate(y_mimic_list))
    y_pre = clf.predict(V_user)

    acc = accuracy_score(y_pre, y_user)
    return acc


def cm_random_selection(model_list, C_indices_list, X_user, y_user, C_indices_user, random_state=0):
    n_developer=len(model_list)
    idx_list=[]
    for i in range(n_developer):
        if C_indices_list[i]==C_indices_user:
            idx_list.append(i)
    np.random.seed(random_state)
    selected_idx=np.random.choice(idx_list, 1)[0]
    y_pre=model_list[selected_idx].predict(X_user)
    acc=accuracy_score(y_pre, y_user)
    return acc


def cm_ensemble_all(model_list, C_indices_list, X_user, y_user, C_indices_user):
    n_developer = len(model_list)
    idx_list = []
    for i in range(n_developer):
        if C_indices_list[i] == C_indices_user:
            idx_list.append(i)

    prediction_dict = dict()
    for idx in idx_list:
        class_list = model_list[idx].classes_
        proba_matrix = model_list[idx].predict_proba(X_user)
        for i in range(len(class_list)):
            key = class_list[i]
            if key not in prediction_dict.keys():
                prediction_dict[key] = proba_matrix[:, i]
            else:
                prediction_dict[key] += proba_matrix[:, i]
    overall_class_list = list(prediction_dict.keys())
    n_class_total = len(overall_class_list)
    final_prediction_proba = np.zeros((X_user.shape[0], n_class_total))
    for i in range(n_class_total):
        final_prediction_proba[:, i] = prediction_dict[overall_class_list[i]]

    indices = np.argmax(final_prediction_proba, axis=1)
    y_p = []
    for idx in indices:
        y_p.append(overall_class_list[idx])
    y_pre = np.array(y_p)
    acc = accuracy_score(y_pre, y_user)
    return acc

def cm_mmd_minimal(model_list, specification_list, C_indices_list, X_user, y_user, C_indices_user):
    n_developer = len(model_list)
    selected_model_list = []
    selected_specification_list = []
    for i in range(n_developer):
        if C_indices_list[i] == C_indices_user:
            selected_model_list.append(model_list[i])
            selected_specification_list.append(specification_list[i])

    mmd_distances=mmd_distance_list(selected_specification_list, X_user)

    selected_idx = np.argmin(mmd_distances)
    y_pre = selected_model_list[selected_idx].predict(X_user)
    acc = accuracy_score(y_pre, y_user)
    return acc

def cm_simplified_previous_work(model_list, specification_list, C_indices_list, X_user, y_user, C_indices_user):
    n_developer = len(model_list)
    selected_model_list = []
    selected_specification_list = []
    for i in range(n_developer):
        if C_indices_list[i] == C_indices_user:
            selected_model_list.append(model_list[i])
            selected_specification_list.append(specification_list[i])

    model_selector_gamma=specification_list[0].gamma

    model_selector=SVC(gamma=model_selector_gamma)
    X=[]
    y=[]
    for idx, selected_specification in enumerate(selected_specification_list):
        temp_samples=selected_specification.z
        X.append(temp_samples)
        y.append([idx]*len(temp_samples))
    X=np.concatenate(X)
    y=np.concatenate(y)

    prediction_list=[]
    for model in selected_model_list:
        y_temp=model.predict(X_user)
        prediction_list.append(y_temp)

    model_selector.fit(X,y)
    model_selection_results=model_selector.predict(X_user)
    final_prediction=[]
    for i in range(len(y_user)):
        final_prediction.append(prediction_list[model_selection_results[i]][i])

    final_prediction = np.array(final_prediction)
    acc = accuracy_score(y_user, final_prediction)
    return acc


def generate_subspace_learning_input(specification_list, C_indices_list, R_indices_list, dim_list):
    n_blocks=len(dim_list)
    specification_block_list=[]
    cardinality_array=[]
    for i in range(len(specification_list)):
        cardinality_array.append(specification_list[i].z.shape[0])
        specification_blocks=[]
        temp_dim_list=[dim_list[j] for j in C_indices_list[i]]
        temp_idx_list=np.cumsum(temp_dim_list)
        temp_idx_list=np.concatenate([[0], temp_idx_list])
        for j in range(len(temp_idx_list)-1):
            specification_blocks.append(specification_list[i].z[:,temp_idx_list[j]:temp_idx_list[j+1]])
        specification_block_list.append(specification_blocks)
    Z_list=[]
    Gamma_list=[]
    for j in range(n_blocks):
        Z_blocks=[]
        Gamma_blocks=[]
        for i in R_indices_list[j]:
            temp_idx=np.argwhere(np.array(C_indices_list[i])==j).squeeze()
            Z_blocks.append(specification_block_list[i][temp_idx])
            Gamma_blocks.append(specification_list[i].beta)
        Z_blocks=np.concatenate(Z_blocks)
        Gamma=np.diag(np.concatenate(Gamma_blocks))
        Z_list.append(Z_blocks.T)
        Gamma_list.append(Gamma)
    return Z_list, Gamma_list, cardinality_array, specification_block_list


def split_X(X, dim_list, C_indices):
    temp_dim_list = [dim_list[j] for j in C_indices]
    temp_idx_list = np.cumsum(temp_dim_list)
    temp_idx_list = np.concatenate([[0], temp_idx_list])
    X_list=[]
    for j in range(len(temp_idx_list) - 1):
        X_list.append(X[:, temp_idx_list[j]:temp_idx_list[j + 1]])
    return X_list

def get_dim_list(X_list):
    dim_list=[]
    for i in range(len(X_list)):
        dim_list.append(X_list[i].shape[1])
    return dim_list

if __name__=='__main__':
    task_name='digits'
    list_generation_type = 'split'
    experiment_remark = 'results (%s)'%list_generation_type
    experiment_name = '%s (%s)' % (task_name, experiment_remark)

    sl_loss_array=[]

    with open(os.path.join('results', experiment_name + '.txt'), 'w') as f:
        if list_generation_type=='split':
            temp_experiment_paras = experiment_paras[task_name]
        else:
            temp_experiment_paras = experiment_supp_paras[task_name]

        print('==Temporary experiment paras==')
        print(temp_experiment_paras)

        output_info=run_learnware_procedure(task_name=task_name, list_generation_type=list_generation_type, **temp_experiment_paras)

        final_results=output_info['final_results']

        sl_loss_array.append(output_info['sl_loss'])

        for i in range(3):
            for j in range(6):
                print('%.3f (%.3f)'%(final_results[j][i][0], final_results[j][i][1]), end='\t')
            print()

        for i in range(3):
            for j in range(6):
                print('%.3f $\pm$ %.3f'%(final_results[j][i][0], final_results[j][i][1]), end='\t')
            print()

        f.write('Extra experiment parameters:\n')
        for key in experiment_paras:
            f.write('%s: %s\n' % (key, experiment_paras[key]))
        f.write('\n')

        f.write('Experiment results:\n')
        for i in range(3):
            for j in range(6):
                f.write('%.3f (%.3f)' % (final_results[j][i][0], final_results[j][i][1]))
                f.write('\t')
            f.write('\n')
        f.write('\n\n')

        f.flush()

    plt.figure()
    for i in range(len(sl_loss_array)):
        plt.plot(sl_loss_array[i]/np.max(sl_loss_array[i]))
    plt.savefig('results/%s (loss figure)'%task_name, dpi=480)