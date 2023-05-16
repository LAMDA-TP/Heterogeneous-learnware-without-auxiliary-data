import numpy as np


def generate_class_assignment_list(class_block):
    assert len(class_block) == 6
    class_assignment_list = [
        np.concatenate([class_block[0], class_block[1]]),
        np.concatenate([class_block[2], class_block[3]]),
        np.concatenate([class_block[2], class_block[5]]),
        np.concatenate([class_block[4], class_block[3]]),
        np.concatenate([class_block[0], class_block[5]]),
        np.concatenate([class_block[4], class_block[1]]),
    ]
    return class_assignment_list


paras_setup = {
    "mfeat_1": {
        "class_assignment_list": generate_class_assignment_list(
            [[0, 6], [1, 7], [2, 8], [3, 9], [4], [5]]
        ),
        "C_indices_list": [[0, 1], [0, 1], [1, 2], [1, 2], [0, 2], [0, 2]],
        "R_indices_list": [[0, 1, 4, 5], [0, 1, 2, 3], [2, 3, 4, 5]],
    },
    "mfeat_2": {
        "class_assignment_list": generate_class_assignment_list(
            [[0, 6], [1, 7], [2, 8], [3, 9], [4], [5]]
        ),
        "C_indices_list": [[0, 1], [0, 1], [1, 2], [1, 2], [0, 2], [0, 2]],
        "R_indices_list": [[0, 1, 4, 5], [0, 1, 2, 3], [2, 3, 4, 5]],
    },
    "digits": {
        "class_assignment_list": generate_class_assignment_list(
            [[0, 6], [1, 7], [2, 8], [3, 9], [4], [5]]
        ),
        "C_indices_list": [[0, 1], [0, 1], [1, 2], [1, 2], [0, 2], [0, 2]],
        "R_indices_list": [[0, 1, 4, 5], [0, 1, 2, 3], [2, 3, 4, 5]],
    },
    "covtype": {
        "class_assignment_list": generate_class_assignment_list(
            [[0], [1], [2], [3], [4], [5]]
        ),
        "C_indices_list": [[0, 1], [0, 1], [1, 2], [1, 2], [0, 2], [0, 2]],
        "R_indices_list": [[0, 1, 4, 5], [0, 1, 2, 3], [2, 3, 4, 5]],
    },
    "kddcup99": {
        "class_assignment_list": generate_class_assignment_list(
            [[0], [1], [2], [3], [4], [5]]
        ),
        "C_indices_list": [[0, 1], [0, 1], [1, 2], [1, 2], [0, 2], [0, 2]],
        "R_indices_list": [[0, 1, 4, 5], [0, 1, 2, 3], [2, 3, 4, 5]],
    },
    "mimic": {
        "class_assignment_list": [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
        "C_indices_list": [[0, 1], [0, 1], [1, 2], [1, 2], [0, 2], [0, 2]],
        "R_indices_list": [[0, 1, 4, 5], [0, 1, 2, 3], [2, 3, 4, 5]],
    },
    "abalone": {
        "class_assignment_list": generate_class_assignment_list(
            [[0, 6], [1, 7], [2], [3], [4], [5]]
        ),
        "C_indices_list": [[0, 1], [0, 1], [1, 2], [1, 2], [0, 2], [0, 2]],
        "R_indices_list": [[0, 1, 4, 5], [0, 1, 2, 3], [2, 3, 4, 5]],
    },
    "awa_1": {
        "class_assignment_list": generate_class_assignment_list(
            [[0, 6], [1, 7], [2, 8], [3, 9], [4], [5]]
        ),
        "C_indices_list": [[0, 1], [0, 1], [1, 2], [1, 2], [0, 2], [0, 2]],
        "R_indices_list": [[0, 1, 4, 5], [0, 1, 2, 3], [2, 3, 4, 5]],
    },
    "awa_2": {
        "class_assignment_list": generate_class_assignment_list(
            [[0, 6], [1, 7], [2, 8], [3, 9], [4], [5]]
        ),
        "C_indices_list": [[0, 1], [0, 1], [1, 2], [1, 2], [0, 2], [0, 2]],
        "R_indices_list": [[0, 1, 4, 5], [0, 1, 2, 3], [2, 3, 4, 5]],
    },
    "mosei": {
        "class_assignment_list": generate_class_assignment_list(
            [[0, 6], [1, 7], [2], [3], [4], [5]]
        ),
        "C_indices_list": [[0, 1], [0, 1], [1, 2], [1, 2], [0, 2], [0, 2]],
        "R_indices_list": [[0, 1, 4, 5], [0, 1, 2, 3], [2, 3, 4, 5]],
    },
    "anuran": {
        "class_assignment_list": generate_class_assignment_list(
            [[0, 6], [1, 7], [2, 8], [3, 9], [4], [5]]
        ),
        "C_indices_list": [[0, 1], [0, 1], [1, 2], [1, 2], [0, 2], [0, 2]],
        "R_indices_list": [[0, 1, 4, 5], [0, 1, 2, 3], [2, 3, 4, 5]],
    },
}

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

experiment_paras = {
    'mfeat_1': {
        'sl_dim': 10,
        'developer_model': RandomForestClassifier(),
    },
    'mfeat_2': {
        'sl_dim': 10,
        'developer_model': RandomForestClassifier(),
        'rkme_gamma': 1e-1,
        'sl_max_iter': 1500,
    },
    'digits': {
        'sl_dim': 10,
        'developer_model': SVC(kernel='linear', probability=True),
    },
    'kddcup99': {
        'sl_dim': 9,
        'developer_model': SVC(probability=True),
    },
    'covtype': {
        'sl_dim': 3,
        'C_indices_user': [0, 2],
        'developer_model': SVC(kernel='linear', probability=True),
        'sl_max_iter': 1500,
    },
    'anuran': {
        'sl_dim': 10,
        'developer_model': SVC(gamma=1e-2, probability=True),
    }
}


experiment_supp_paras = {
    'digits': {
        'sl_dim': 10,
        'developer_model': SVC(kernel='linear', probability=True),
    },
    'kddcup99': {
        'sl_dim': 9,
        'sl_beta': 1e1,
        'developer_model': SVC(kernel='linear', probability=True),
        'sl_max_iter': 1500,
    },
    'covtype': {
        'sl_dim': 6,
        'sl_beta': 1e1,
        'developer_model': SVC(kernel='linear', probability=True),
        'sl_max_iter': 1500,
    },
    'anuran': {
        'sl_dim': 10,
        'developer_model': SVC(gamma=1e-2, probability=True),
    }
}