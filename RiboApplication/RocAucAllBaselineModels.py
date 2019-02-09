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
# import matplotlib.pyplot as plt
import pickle
# import roc

plt.figure()
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

def calculate_roc(y_test, y_score):
    n_classes = 24
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    print ("ROC_AUC")
    print (roc_auc)
    colors = cycle(['blue', 'red', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data')
    plt.legend(loc="lower right")
    plt.show()

def construct_models(X_train, X_test, y_train, y_test, y_test_bin):
    confusion_matrices={}
    classifiers = [
        # AdaBoostClassifier(),
        # GaussianNB()  
        # KNeighborsClassifier(),
        # DecisionTreeClassifier(),
        RandomForestClassifier(),
        # OneVsRestClassifier(MLPClassifier(max_iter=100))    
        ]
    for clf in  classifiers:
        model = clf.fit(X_train, y_train)
        # print("classifier", model)
        # print ("Accuracy on Train Set")
        # print (model.score(X_train, y_train))
        print ("Accuracy on Test Set")
        print (model.score(X_test, y_test))        
        # print ("Report")
        # print (classification_report(y_test,model.predict(X_test))) 
        # print ("Confusion Matrix")
        y_score = model.predict_proba(X_test)
        # confusion_matrices[str(clf)] = confusion_matrix(y_test,model.predict(X_test))
        # roc_and_auc(confusion_matrices[str(clf)])
        # print (confusion_matrix(y_test,model.predict(X_test))) 
    calculate_roc(y_test_bin, y_score)


def roc_and_auc(confusion_matrix_for_a_model):
    print ("GG")

Data = []
Output = []
bin_output = []

#Exporting the CSV paths
#Path = 'datasets/NN/16_riboswitches.csv'
Path = 'datasets/NN/24_riboswitches.csv'

#Call function to Load Dataset
Data, Output = Create_Data(Path, Data, Output)

#Converting the train data into Float
Data, Output = Convert_to_Float(Data, Output)

#Divide Dataset for training and testing
Data_train, Data_test, Output_train, Output_test = train_test_split(Data, Output, test_size=0.2, stratify=Output)

unique_classes = list(set(Output))
unique_classes.sort()
bin_output = label_binarize(Output_test, classes=unique_classes)

# #Converting the train data into Float
# Data_train, Output_train = Convert_to_Float(Data_train, Output_train)


#Preprocessing the data
scaler = StandardScaler()
scaler.fit(Data_train)
Data_train = scaler.transform(Data_train)
Data_test = scaler.transform(Data_test)

construct_models(Data_train, Data_test, Output_train, Output_test, bin_output)

