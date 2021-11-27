import matplotlib.pyplot as plt
from Hyper_parameters import HyperParams
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from myDataLoader import get_ndarrays
import numpy as np

def main():
    x_train, y_train, x_test, y_test, x_valid, y_valid = get_ndarrays(test=True, flatten=True)
    # mel: 128 x 128
    SVM = SVC(kernel='linear')
    SVM.fit(x_train, y_train)
    y_pred = SVM.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test, y_pred), recall_score(y_test, y_pred, average='macro'), precision_score(y_test, y_pred, average='macro'))

        
if __name__ == '__main__':
    main()