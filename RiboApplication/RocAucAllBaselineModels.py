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

# import matplotlib.pyplot as #plt
import pickle
# import roc

#plt.figure()
lw = 2

#Load Dataset
def Create_Data(Path, Data, Output):
    with open(Path) as csvfile:
        Data_Path = list(csv.DictReader(csvfile))
        for x in Data_Path:
                #Creating the feature vector of mono and di nucleotides
                Data.append([x["A"], x["T"], x["G"], x["C"],x["AA"], x["AC"], x["AG"], x["AU"],x["CA"], x["CC"], x["CG"], x["CU"],x["GA"], x["GC"], x["GG"], x["GU"],x["UA"], x["UC"], x["UG"], x["UU"]])
                Output.append(x["Type"])
        return Data, Output

#Converting the values to Float for Mathematical purposes
def Convert_to_Float(Data, Output):
    for i in range(len(Data)):
        for j in range(20):
            Data[i][j]=float(Data[i][j])
        Output[i]= int(Output[i])
    return Data, Output


def calculate_roc(y_test, y_score, name):
    each_class = []
    n_classes = 24
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # print ("ROC_AUC")
    # print (roc_auc)

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
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
        #plt.plot(fpr[i], tpr[i], color=color, lw=lw,label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
        each_class.append(round(roc_auc[i], 2))
    #plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    #plt.xlim([-0.05, 1.0])
    #plt.ylim([0.0, 1.05])
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.title('Receiver operating characteristic for multi-class data : ' + name)
    #plt.legend(loc="lower right")
    #plt.show()
    return each_class

def create_aoc_table(overall):
    overall = np.array(overall)
    df = pd.DataFrame(overall.T)
    print ("The generated dataframe")
    print (df)
    # writer = pd.ExcelWriter('excelTables/PythonExport.xlsx')
    # df.to_excel(writer,'Sheet1')
    # writer.save() 


def construct_models(X_train, X_test, y_train, y_test, y_test_bin):
    confusion_matrices={}
    classifiers = [
        # AdaBoostClassifier(),
        # GaussianNB()  
        # KNeighborsClassifier(),
        # DecisionTreeClassifier(),
        # RandomForestClassifier(),
        AdaBoostClassifier(),
        GaussianNB(),  
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        MLPClassifier()  
        # OneVsRestClassifier(MLPClassifier(max_iter=100))    
        ]
    names = ["AdaBoostClassifierModel","GaussianNBModel","KNeighborsClassifierModel","DecisionTreeClassifierModel","RandomForestClassifierModel","MLPClassifierModel"]
    for clf,name in  zip(classifiers,names):
        model = clf.fit(X_train, y_train)
        # print("classifier", model)
        # print ("Accuracy on Train Set")
        # print (model.score(X_train, y_train))
        print (name)
        print ("Accuracy on Test Set")
        print (model.score(X_test, y_test))  
        filename = 'pickled_models/' + name + '.pkl'
        pickle.dump(model, open(filename, 'wb'))        
        # print ("Report")
        # print (classification_report(y_test,model.predict(X_test))) 
        # print ("Confusion Matrix")
        y_score = model.predict_proba(X_test)
        # confusion_matrices[str(clf)] = confusion_matrix(y_test,model.predict(X_test))
        # roc_and_auc(confusion_matrices[str(clf)])
        # print (confusion_matrix(y_test,model.predict(X_test))) 

def generate_roc(X_train, X_test, y_train, y_test, y_test_bin):
    confusion_matrices={}
    classifiers = [
        AdaBoostClassifier(),
        GaussianNB(),  
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        MLPClassifier()   
        ]
    names = ["AdaBoostClassifierModel","GaussianNBModel","KNeighborsClassifierModel","DecisionTreeClassifierModel","RandomForestClassifierModel","MLPClassifierModel"]
    overall = []
    for clf,name in  zip(classifiers,names):
        filename = 'pickled_models/' + name + '.pkl'
        model = pickle.load(open(filename, 'rb'))
        # print ("Accuracy on Test Set : " + name)
        # print (model.score(X_test, y_test))         
        # print ("Report : " + name)
        # print (y_test)
        # print (model.predict(X_test))
        # print (classification_report(y_test,model.predict(X_test))) 

        # print ("Confusion Matrix")
        # confusion_matrices[str(clf)] = confusion_matrix(y_test,model.predict(X_test))
        # roc_and_auc(confusion_matrices[str(clf)])
        # print (confusion_matrix(y_test,model.predict(X_test)))
        y_score = model.predict_proba(X_test) 
        each_class = calculate_roc(y_test_bin, y_score, name)
        overall.append(each_class)
    create_aoc_table(overall)

def roc_and_auc(confusion_matrix_for_a_model):
    print ("GG")

Data_train = []
Output_train = []
Data_test = []
Output_test = []
bin_output = []

#Exporting the CSV paths
#Path = 'datasets/NN/16_riboswitches.csv'
# Path = 'datasets/NN/24_riboswitches.csv'
Path = 'processed_datasets/24_riboswitches_final_train.csv'

#Call function to Load Dataset
Data_train, Output_train = Create_Data(Path, Data_train, Output_train)

#Converting the train data into Float
Data_train, Output_train = Convert_to_Float(Data_train, Output_train)

Path = 'processed_datasets/24_riboswitches_final_test.csv'

#Call function to Load Dataset
Data_test, Output_test = Create_Data(Path, Data_test, Output_test)

#Converting the train data into Float
Data_test, Output_test = Convert_to_Float(Data_test, Output_test)

#Divide Dataset for training and testing
#Data_train, Data_test, Output_train, Output_test = train_test_split(Data, Output, test_size=0.1, stratify=Output)

# print (Data_test)
# print (Output_test)
unique_classes = list(set(Output_test))
unique_classes.sort()
print (unique_classes)
bin_output = label_binarize(Output_test, classes=unique_classes)

# #Converting the train data into Float
# Data_train, Output_train = Convert_to_Float(Data_train, Output_train)


#Preprocessing the data
scaler = StandardScaler()
scaler.fit(Data_train)
Data_train = scaler.transform(Data_train)
Data_test = scaler.transform(Data_test)

# construct_models(Data_train, Data_test, Output_train, Output_test, bin_output)
generate_roc(Data_train, Data_test, Output_train, Output_test, bin_output)

# Accuracy on Test Set
# 0.395582660293
# Accuracy on Test Set
# 0.540959287951
# Accuracy on Test Set
# 0.798912147684
# Accuracy on Test Set
# 0.706609526949
# C:\Users\Welcome\AppData\Local\conda\conda\envs\kesh1\lib\site-packages\sklearn\ensemble\forest.py:248: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
#   "10 in version 0.20 to 100 in 0.22.", FutureWarning)
# Accuracy on Test Set
# 0.782429536839
# C:\Users\Welcome\AppData\Local\conda\conda\envs\kesh1\lib\site-packages\sklearn\neural_network\multilayer_perceptron.py:562: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
#   % self.max_iter, ConvergenceWarning)
# Accuracy on Test Set
# 0.844074501401

