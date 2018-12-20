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

    return [TP,FN,AP,FP,TN,AN]

True_Positives={}
False_Negatives={}
All_Positives={}
False_Positives={}
True_Negatives={}
All_Negatives={}
for i,j in confu.items():
    True_Positives[i]=metrics(j)[0]
    False_Negatives[i]=metrics(j)[1]
    All_Positives[i]=metrics(j)[2]
    False_Positives[i]=metrics(j)[3]
    True_Negatives[i]=metrics(j)[4]
    All_Negatives[i]=metrics(j)[5]


def Rec(x,y,average="True"):
    Recal={}
    names=[]
    recal=[]
    a=[]
    b=[]
    for i,j in x.items():
        names.append(i)
        a.append(j)
    for i,j in y.items():
        b.append(j)
    for i,j in zip(a,b):
        rec=[]
        for a,b in zip(i,j):
            recall = float(a)/float(b)
            rec.append(recall)
        recal.append(rec)
    if average=="False":
        for i,j in zip(names,recal):
            Recal[i]=j
    else:
        sum_recall=0
        avg_recall=[]
        for i in recal:

            sum_recall=0
            for j in i:

                sum_recall+=j
            avg_recall.append(sum_recall/len(i))
        for i,j in zip(names,avg_recall):
            Recal[i.split('C')[0]]=j

    return Recal
def Pre(x,y,average="True"):
    Recal={}
    names=[]
    recal=[]
    a=[]
    b=[]
    for i,j in x.items():
        names.append(i)
        a.append(j)
    for i,j in y.items():
        b.append(j)
    for i,j in zip(a,b):
        rec=[]
        for a,b in zip(i,j):
            recall = float(a)/(float(a)+float(b))
            rec.append(recall)
        recal.append(rec)
    if average=="False":
        for i,j in zip(names,recal):
            Recal[i]=j
    else:
        sum_recall=0
        avg_recall=[]
        for i in recal:

            sum_recall=0
            for j in i:

                sum_recall+=j
            avg_recall.append(sum_recall/len(i))
        for i,j in zip(names,avg_recall):
            Recal[i.split('C')[0]]=j

    return Recal

def Acc(w,x,y,z,average="True"):
    Recal={}
    names=[]
    recal=[]
    a=[]
    b=[]
    c=[]
    d=[]
    for i,j in w.items():
        names.append(i)
        a.append(j)
    for i,j in x.items():
        b.append(j)
    for i,j in y.items():
        c.append(j)
    for i,j in z.items():
        d.append(j)
    for i,j,k,l in zip(a,b,c,d):
        rec=[]
        for a,b,c,d in zip(i,j,k,l):
            recall = (float(a)+float(b))/(float(a)+float(b)+float(c)+float(d))
            rec.append(recall)
        recal.append(rec)
    if average=="False":
        for i,j in zip(names,recal):
            Recall[i]=j
    else:
        sum_recall=0
        avg_recall=[]
        for i in recal:

            sum_recall=0
            for j in i:

                sum_recall+=j
            avg_recall.append(sum_recall/len(i))
        for i,j in zip(names,avg_recall):
            Recal[i.split('C')[0]]=j

    return Recal

def F(x,y,average="True"):
    Recal={}
    names=[]
    recal=[]
    a=[]
    b=[]
    for i,j in x.items():
        names.append(i)
        a.append(j)
    for i,j in y.items():
        b.append(j)
    for i,j in zip(a,b):
        rec=[]
        for a,b in zip(i,j):
            recall = (2*float(a)*float(b))/(float(a)+float(b))
            rec.append(recall)
        recal.append(rec)
    if average=="False":
        for i,j in zip(names,recal):
            Recal[i]=j
    else:
        sum_recall=0
        avg_recall=[]
        for i in recal:

            sum_recall=0
            for j in i:

                sum_recall+=j
            avg_recall.append(sum_recall/len(i))
        for i,j in zip(names,avg_recall):
            Recal[i.split('C')[0]]=j

    return Recal

def fdr(x,y,average="True"):
    Recal={}
    names=[]
    recal=[]
    a=[]
    b=[]
    for i,j in x.items():
        names.append(i)
        a.append(j)
    for i,j in y.items():
        b.append(j)
    for i,j in zip(a,b):
        rec=[]
        for a,b in zip(i,j):
            recall = float(a)/float(b)
            rec.append(recall)
        recal.append(rec)
    if average=="False":
        for i,j in zip(names,recal):
            Recal[i]=j
    else:
        sum_recall=0
        avg_recall=[]
        for i in recal:

            sum_recall=0
            for j in i:

                sum_recall+=j
            avg_recall.append(sum_recall/len(i))
        for i,j in zip(names,avg_recall):
            Recal[i.split('C')[0]]=j

    return Recal
Recall=Rec(True_Positives,All_Positives)
Precision=Pre(True_Positives,False_Positives)
Accuracy=Acc(True_Positives,True_Negatives,False_Positives,False_Negatives)
Precisionf=Pre(True_Positives,False_Positives,average='False')
Recallf=Rec(True_Positives,All_Positives,average='False')
F1=F(Precisionf,Recallf)
FPR=fdr(False_Positives,All_Negatives)












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
plt.xticks(np.arange(0,0.01,0.01))
ax.set_title('FPR for the different classifiers')

plt.show()
