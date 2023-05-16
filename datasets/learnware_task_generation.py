"""
This file is used to generate the datasets of developers and the user in the learnware scenario
"""

from collections import Counter
import numpy as np

from sklearn.model_selection import train_test_split
from tools.clf_performance import clf_performance_test

class LearnwareScenarioGenerator():

    def __init__(self, X_list, y, class_assignment_list, random_state=0):
        self._check_y(y)

        self.n_views=len(X_list)
        self.n_classes=len(set(y))
        self.dim_list=self.get_dim_list(X_list)
        self.random_state=random_state

        self.initialization(X_list, y, class_assignment_list)

    def _check_y(self, y):
        """ check whether the label set is [0,1,...,n_classes-1]

        Args:
            y:

        Returns:

        """
        if set(y) != set(range(len(set(y)))):
            raise Exception('illegal labels: %s' % sorted(Counter(y)))

    def get_dim_list(self, X_list):
        """

        Args:
            X_list: [n*d1, n*d2, ..., n*dk]

        Returns:

        """
        dim_list=[]
        for i in range(self.n_views):
            dim_list.append(X_list[i].shape[1])
        return dim_list

    def initialization(self, X_list, y, class_assignment_list):
        X=np.concatenate(X_list, axis=1)
        X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, stratify=y)
        X_list_train, y_list_train=self._split_data_by_labels(X_train, y_train)
        X_list_test, y_list_test=self._split_data_by_labels(X_test, y_test)
        X_candidate_list_train, y_candidate_list_train=self.aggregate_data(X_list_train, y_list_train, class_assignment_list)
        X_candidate_list_test, y_candidate_list_test = self.aggregate_data(X_list_test, y_list_test, class_assignment_list)
        self.X_candidate_list_train=X_candidate_list_train
        self.y_candidate_list_train=y_candidate_list_train
        self.X_candidate_list_test = X_candidate_list_test
        self.y_candidate_list_test = y_candidate_list_test

    def aggregate_data(self, X_list, y_list, class_assignment_list):
        X_candidate_list = []
        y_candidate_list = []
        for class_assignment in class_assignment_list:
            X_list_temp = []
            y_list_temp = []
            for idx in class_assignment:
                X_list_temp.append(X_list[idx])
                y_list_temp.append(y_list[idx])
            X_candidate=np.concatenate(X_list_temp)
            y_candidate=np.concatenate(y_list_temp)
            X_candidate_list.append(X_candidate)
            y_candidate_list.append(y_candidate)
        return X_candidate_list, y_candidate_list

    def _split_data_by_labels(self, X, y):
        label_set = list(sorted(Counter(y)))
        X_list_per_class = []
        y_per_class = []
        for label in label_set:
            idx_temp=np.argwhere(y==label).squeeze()
            X_list_per_class.append(X[idx_temp,:])
            y_per_class.append(y[idx_temp])
        return X_list_per_class, y_per_class

    def _split_data_by_features(self, X, dim_list):
        """ split data into multiple features

        Args:
            X:
            dim_list:

        Returns:

        """
        X_list=[]
        idx_list=np.cumsum(dim_list)
        idx_list=np.concatenate([[0], idx_list])
        for i in range(len(idx_list)-1):
            X_temp=X[:,idx_list[i]:idx_list[i+1]]
            X_list.append(X_temp)
        return X_list

    def generate_developers(self, C_indices_list):
        """

        Returns:

        """
        assert len(self.X_candidate_list_train)==len(C_indices_list)
        assert len(self.y_candidate_list_train)==len(C_indices_list)

        X_list_dev=[]
        y_list_dev=[]
        for i, C_indices in enumerate(C_indices_list):
            X=self.X_candidate_list_train[i]
            y=self.y_candidate_list_train[i]
            X_list_all=self._split_data_by_features(X, self.dim_list)
            X_list_temp=[]
            for j in C_indices:
                X_list_temp.append(X_list_all[j])
            X_dev=np.concatenate(X_list_temp, axis=1)
            X_list_dev.append(X_dev)
            y_list_dev.append(y)
        return X_list_dev, y_list_dev

    def generate_user(self, C_indices, selected_task_indices):
        X_list=[]
        y_list=[]
        for idx in selected_task_indices:
            X_list.append(self.X_candidate_list_test[idx])
            y_list.append(self.y_candidate_list_test[idx])
        X=np.concatenate(X_list)
        y=np.concatenate(y_list)
        X_list_all=self._split_data_by_features(X, self.dim_list)
        X_list_selected=[]
        for idx in C_indices:
            X_list_selected.append(X_list_all[idx])
        X_user=np.concatenate(X_list_selected, axis=1)
        return X_user, y


def generate_class_assignment_list(y, n_learnwares, verbose=0):
    label_set=list(sorted(Counter(y)))
    class_assignment_list=[]
    for _ in range(n_learnwares):
        class_assignment_list.append([])
    for idx, label in enumerate(label_set):
        class_assignment_list[idx % n_learnwares].append(label)
    if verbose>=1:
        print('class assignment list:', class_assignment_list)
    return class_assignment_list


def describe_learnware_scenario_market(X_list_dev, y_list_dev, clf):
    assert len(X_list_dev)==len(y_list_dev)

    n_developers=len(X_list_dev)
    for i in range(n_developers):
        acc_mean, acc_std, _=clf_performance_test(clf, X_list_dev[i], y_list_dev[i], verbose=0)
        print('developer %d | fea: %s | label: %s | label set: %s | acc cv results: %.3f (%.3f)'%(i,X_list_dev[i].shape, len(y_list_dev[i]), sorted(Counter(y_list_dev[i])), acc_mean, acc_std))


def describe_learnware_scenario_user(X_user, y_user, clf):
    acc_mean, acc_std, _=clf_performance_test(clf, X_user, y_user, verbose=0)
    print('user | fea: %s | label: %s | label set: %s | acc cv results: %.3f (%.3f)'%(X_user.shape, len(y_user), sorted(Counter(y_user)), acc_mean, acc_std))


if __name__=='__main__':
    from benchmarks import load_dataset_multi_fea, describe_dataset

    verbose = 1
    dataset_name = 'mfeat'
    X_list, y = load_dataset_multi_fea(dataset_name, verbose)
    describe_dataset(X_list, y, dataset_name)

    class_assignment_list=generate_class_assignment_list(y, 6, 1)
    generator=LearnwareScenarioGenerator(X_list[0:3], y, class_assignment_list)
    C_indices_list=[[0,1],[0,1],[1,2],[1,2],[0,2],[0,2]]
    X_list_dev, y_list_dev=generator.generate_developers(C_indices_list)
    X_user, y = generator.generate_user([0], [0])
    print()