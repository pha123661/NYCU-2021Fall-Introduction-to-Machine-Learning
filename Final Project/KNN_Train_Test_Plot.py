import matplotlib.pyplot as plt
from Hyper_parameters import HyperParams
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from myDataLoader import get_ndarrays
import numpy as np

def main():
    x_train, y_train, x_test, y_valid, x_valid, y_test = get_ndarrays(test=True, flatten=True)
    # mel: 128 x 128
    KNNclf = KNeighborsClassifier(n_neighbors=2)
    KNNclf.fit(x_train, y_train)
    y_pred = KNNclf.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test, y_pred), recall_score(y_test, y_pred, average='macro'), precision_score(y_test, y_pred, average='macro'))

        
if __name__ == '__main__':
    main()