import csv
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from scipy import interp
import pickle
import pandas as pd
from scipy import interp
from openpyxl.workbook import Workbook

#plt.figure()
lw = 2

def create_aoc_table(overall,name):
    overall = np.array(overall)
    df = pd.DataFrame(overall.T)
    print ("The generated dataframe")
    print (df)
    writer = pd.ExcelWriter('excelTables/modelsAocValuesTableCNN.xlsx')
    df.to_excel(writer,name)
    writer.save() 

def calculate_roc(y_test, y_score, name):
    each_class = []
    unique_classes = list(set(y_test))
    unique_classes.sort()
    print (unique_classes)
    bin_output = label_binarize(y_test, classes=unique_classes)
    n_classes = 24
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(bin_output[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # print ("ROC_AUC")
    # print (roc_auc)

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(bin_output.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.plot(0, 0,
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='white', linestyle=':', linewidth=4)

    #plt.plot(fpr["micro"], tpr["micro"],
            # label='micro-average ROC curve (area = {0:0.2f})'
            #     ''.format(roc_auc["micro"]),
            # color='deeppink', linestyle=':', linewidth=4)

    #plt.plot(fpr["macro"], tpr["macro"],
            # label='macro-average ROC curve (area = {0:0.2f})'
            #     ''.format(roc_auc["macro"]),
            # color='navy', linestyle=':', linewidth=4)

    colors = cycle([
        '#aa65bb', '#c8a581', '#701f57','#f5aed0', '#7288ee', '#f6bcba',
        '#6d4018', '#44cbe9', '#f48a2a','#2efb0e', '#aeee77', '#0e4967',
        '#257d9d','#2c0ec4','#441401','#6b3ae9','#576377','#18713a',
        '#357ad1','#5e8282','#fc0525','#120c63','#FF5733'
    ])
    each_class.append(round(roc_auc["micro"], 2))
    each_class.append(round(roc_auc["macro"], 2))
    for i, color in zip(range(n_classes), colors):
        # plt.plot(fpr[i], tpr[i], color=color, lw=lw,label='ROC curve of class {0} (area = {1:0.2f})'.format(i+1, roc_auc[i]))
        plt.plot(fpr[i], tpr[i], color=color, lw=lw)
        # print (#plt)
        each_class.append(round(roc_auc[i], 2))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data : ' + name)
    plt.legend(loc="lower right")
    plt.show()
    # create_aoc_table(each_class, name)
    print ("GG")