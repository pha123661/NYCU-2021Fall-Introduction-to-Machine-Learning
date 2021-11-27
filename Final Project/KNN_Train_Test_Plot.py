import matplotlib.pyplot as plt
from Hyper_parameters import HyperParams
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from myDataLoader import get_ndarrays
from sklearn.model_selection import GridSearchCV
import numpy as np
from CNN_Train_Test_Plot import *

def main():
    x_train, y_train, x_test, y_test, x_valid, y_valid = get_ndarrays()
    # shape = (samples, 512*2*2)
    KNNclf = KNeighborsClassifier(n_jobs=-1, n_neighbors=7) # get by grid search
    KNNclf.fit(x_train, y_train)
    y_pred = KNNclf.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test, y_pred), recall_score(y_test, y_pred, average='macro'), precision_score(y_test, y_pred, average='macro'))

        
if __name__ == '__main__':
    main()