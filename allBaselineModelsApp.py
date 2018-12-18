import csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import pickle
import numpy as np

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
        Output[i]=float(Output[i])
    return Data, Output

def construct_models(X_train, X_test, y_train, y_test):
    classifiers = [
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        MLPClassifier(),
        ]
    confusion_matrices={}
    for clf in  classifiers:
        model = clf.fit(X_train, y_train)
        #print("classifier", model)
        """
        print ("Accuracy on Train Set")
        print (model.score(Data_train, Output_train))
        print ("Accuracy on Test Set")
        print (model.score(Data_test, Output_test))
        print ("Report")

        print ("Confusion Matrix")
        """
        #print (classification_report(y_test,model.predict(X_test)))
        confusion_matrices[str(clf)]=confusion_matrix(y_test,model.predict(X_test))
    return confusion_matrices

Data = []
Output = []

#Exporting the CSV paths
#Path = 'datasets/NN/16_riboswitches.csv'
Path = 'datasets/NN/24_riboswitches.csv'

#Call function to Load Dataset
Data, Output = Create_Data(Path, Data, Output)

#Divide Dataset for training and testing
Data_train, Data_test, Output_train, Output_test = train_test_split(Data, Output, test_size=0.2)

#Converting the train data into Float
Data_train, Output_train = Convert_to_Float(Data_train, Output_train)

#Converting the test data into Float
Data_test, Output_test = Convert_to_Float(Data_test, Output_test)

#Preprocessing the data
scaler = StandardScaler()
scaler.fit(Data_train)
Data_train = scaler.transform(Data_train)
Data_test = scaler.transform(Data_test)

confu=construct_models(Data_train, Data_test, Output_train, Output_test)



def metrics(matrix):
    i=0
    TP=[]
    FN=[]
    AP=[]
    FP=[]
    TN=[]
    AN=[]
    Recall=[]                  #Recall is the true positive rate
    FPR=[]
    Precision=[]
    Accuracy=[]
    F1=[]
    sum_matrix=0
    for j in range(len(matrix)):
        for k in range(len(matrix)):
            sum_matrix+=matrix[j][k]
    #print("sum matrix:{}".format(sum_matrix))
    while i < len(matrix):
        tp=matrix[i][i]
        TP.append(tp)
        sum = 0
        for j in range(len(matrix[i])):
            sum+=matrix[i][j]
        fn = sum - tp
        AP.append(sum)
        FN.append(fn)
        sum_col = 0
        for j in range(len(matrix[i])):
            sum_col+=matrix[j][i]
        fp = sum_col - tp
        FP.append(fp)
        tn = sum_matrix - tp - fp - fn
        TN.append(tn)
        an = fp + tn
        AN.append(tn)
        i=i+1
    for k,j in zip(TP,AP):
        recall = float(k)/float(j)
        Recall.append(recall)
    for k,j in zip(FP,AN):
        fpr= float(k)/float(j)
        FPR.append(fpr)
    for k,j in zip(TP,FP):
        precision = float(k)/(float(k)+float(j))
        Precision.append(precision)
    for a,b,c,d in zip(TP,TN,FP,FN):
        accuracy= (float(a)+float(b))/(float(a)+float(b)+float(c)+float(d))
        Accuracy.append(accuracy)
    for a,b in zip(Precision,Recall):
        f1=(2*a*b)/(a+b)
        F1.append(f1)
    sum_precision=0
    for j in Precision:
        sum_precision+=j
    avg_precision=sum_precision/len(Precision)
    sum_recall=0
    for j in Recall:
        sum_recall+=j
    avg_recall=sum_recall/len(Recall)
    sum_FPR=0
    for j in FPR:
        sum_FPR+=j
    avg_FPR=sum_FPR/len(FPR)
    sum_accuracy=0
    for j in Accuracy:
        sum_accuracy+=j
    avg_accuracy=sum_accuracy/len(Accuracy)
    sum_F1=0
    for j in F1:
        sum_F1+=j
    avg_F1=sum_F1/len(F1)


    return [avg_precision,avg_recall,avg_FPR,avg_accuracy,avg_F1]
Precision={}
Recall={}
FPR={}
Accuracy={}
F1={}
for i,l in confu.items():
    #dict_metrics[i.split('(')[0]]=metrics(l)
    Precision[i.split('C')[0]]=metrics(l)[0]
    Recall[i.split('C')[0]]=metrics(l)[1]
    FPR[i.split('C')[0]]=metrics(l)[2]
    Accuracy[i.split('C')[0]]=metrics(l)[3]
    F1[i.split('C')[0]]=metrics(l)[4]
    #print("Precision: {}\t Recall: {}\t FDR: {}\t Accuracy: {}\t F1: {}\n".format(metrics(l[0]),metrics(l[1]),metrics(l[2]),metrics(l[3]),metrics(l[4])))
#Precision, Recall, FPR, Accuracy, F1
metric=[]
metric.append(Precision.copy())
metric.append(Recall.copy())
metric.append(Accuracy.copy())
metric.append(F1.copy())
params=['Precision','Recall','Accuracy','F1']
for i,j in zip(metric,params):
    plt.rcdefaults()
    fig, ax = plt.subplots()
    algorithms=list(i.keys())
    y_pos = np.arange(len(algorithms))
    Precison=list(i.values())
    ax.barh(y_pos, Precison, align='center',color='green')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(algorithms,rotation=55)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel(j)
    plt.xticks(np.arange(0,1,0.1))
    ax.set_title('{} for the different classifiers'.format(j))

    plt.show()

    #  create the figure
plt.rcdefaults()
fig, ax = plt.subplots()
algorithms=list(FPR.keys())
y_pos = np.arange(len(algorithms))
Precison=list(FPR.values())
ax.barh(y_pos, Precison, align='center',color='green')
ax.set_yticks(y_pos)
ax.set_yticklabels(algorithms,rotation=55)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel(j)
plt.xticks(np.arange(0,0.01,0.002))
ax.set_title('FPR for the different classifiers')

plt.show()


#print(metrics(mat))
#Classification using tanh activation function
# rf = MLPClassifier()
# mlp = MLPClassifier(solver='sgd',alpha=0.05,learning_rate='adaptive', hidden_layer_sizes=(10, 10), max_iter=500,momentum=0.99, activation='relu', verbose=False)

#Training the model using Training Data
# rf.fit(Data_train, Output_train)
# mlp.fit(Data_train, Output_train)

#Save the model to disk
#filename = 'models/NN/mlp_16_model.plk'
# filename = 'models/NN/mlp_24_model.plk'
# pickle.dump(mlp, open(filename, 'wb'))

# Accuracy on Train and Test Sets
# print ("Accuracy Report")
# print ("Accuracy on Train Set")
# print (rf.score(Data_train, Output_train))
# print ("Accuracy on Test Set")
# print (rf.score(Data_test, Output_test))

# #Predicting the Output using the Test Data
# predictions = rf.predict(Data_test)

# #Printing the Predictions and the Output Labels
# #print(predictions, Output_test)

# #Confusion Matrix
# print ("Confusion Matrix")
# print (confusion_matrix(Output_test,predictions))

# #Classification Report
# print ("Classification Report")
# print (classification_report(Output_test,predictions))
