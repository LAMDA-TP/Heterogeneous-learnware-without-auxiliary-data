import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


def clf_performance_test(clf, X, y, X_test=None, y_test=None, scaler=None, task_name='', verbose=1):
    if scaler is not None:
        scaler.fit(X)
        X = scaler.transform(X)
        if X_test is not None and y_test is not None:
            X_test = scaler.transform(X_test)

    tic = time.time()
    scores = cross_val_score(clf, X, y, cv=5)
    toc = time.time()
    work_time = toc - tic
    cv_mean=scores.mean()
    cv_std=scores.std()
    ood_mean=0
    if verbose>=1:
        print("Accuracy on %s: %0.4f (%0.4f) | time: %.2f s (%.2f min)" % (
            task_name + " (self test)", cv_mean, cv_std, work_time, work_time / 60))

    if X_test is not None and y_test is not None:
        tic = time.time()
        clf.fit(X, y)
        y_pre = clf.predict(X_test)
        score = accuracy_score(y_test, y_pre)
        toc = time.time()
        work_time = toc - tic
        ood_mean=score
        if verbose>=1:
            print("Accuracy on %s: %0.4f | time: %.2f s (%.2f min)" % (
            task_name + " (using extra test data)", ood_mean, work_time, work_time / 60))

    return cv_mean,cv_std,ood_mean