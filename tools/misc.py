import time
import numpy as np

class TimeRecorder(object):

    def __init__(self):
        self.tic=time.time()

    def get_time_str(self):
        toc=time.time()
        duration=toc-self.tic
        return 'Time stamp: %.2f s (%.2f min)'%(duration, duration/60)


def print_mean_std_matrix(mean_mat, std_mat, each_row_name, each_col_name):
    assert mean_mat.shape[0]==std_mat.shape[0]
    assert mean_mat.shape[1]==mean_mat.shape[1]

    print('       	', end='\t')
    for j in range(len(each_col_name)):
        if isinstance(each_col_name[j],str):
            print(f'{each_col_name[j]:^13s}', end='\t')
        else:
            print(f'{each_col_name[j]:^13f}', end='\t')
    print('')
    for i in range(len(each_row_name)):
        if isinstance(each_row_name[i],str):
            print(f'{each_row_name[i]:^8s}', end='\t')
        else:
            print(f'{each_row_name[i]:^8f}', end='\t')
        for j in range(len(each_col_name)):
            print(('%.3f (%.3f)' % (mean_mat[i][j], std_mat[i][j])), end='\t')
        print('')


def label2indicator(labels,uploader_class_assignment):
    n_samples=len(labels)
    n_uploaders=len(uploader_class_assignment)
    indicators=[]
    for i in range(n_samples):
        for j in range(n_uploaders):
            if labels[i] in uploader_class_assignment[j]:
                indicators.append(j)
                break
    return np.array(indicators)