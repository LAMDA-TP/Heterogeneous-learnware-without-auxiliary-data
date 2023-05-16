import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder


def get_anuran_dataset(data_home='benchmarks/anuran', class_type='species'):
    """

    Args:
        data_home:
        class_type: {'family', 'genus', 'species'}

    Returns:

    """
    df = pd.read_csv(os.path.join(data_home,'Frogs_MFCCs.csv'))   # data can be download from https://www.kaggle.com/code/aryan19/anuran-cells-classification
    df = df.drop('RecordID', axis=1)

    C1 = df['Family'].values  # Family
    C2 = df['Genus'].values  # Genus
    C3 = df['Species'].values  # Species
    x = np.array(df.select_dtypes('float64').values)  # Rest of the features

    le = LabelEncoder()
    C1 = np.array(le.fit_transform(C1))
    C2 = np.array(le.fit_transform(C2))
    C3 = np.array(le.fit_transform(C3))

    if class_type=='family':
        return x, C1
    elif class_type=='genus':
        return x, C2
    elif class_type=='species':
        return x, C3
    else:
        raise Exception('invalid class type')


from sklearn.svm import SVC
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


if __name__=='__main__':
    X, y=get_anuran_dataset()
    scaler=MaxAbsScaler()
    X=scaler.fit_transform(X)
    X_train, X_test, y_train, y_test=train_test_split(X,y)
    clf = SVC()
    clf.fit(X_train, y_train)
    y_p = clf.predict(X_test)

    print(accuracy_score(y_p, y_test))