import csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
# import matplotlib.pyplot as plt
import pickle
import roc

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
    confusion_matrices={}
    classifiers = [
        # AdaBoostClassifier(),
        # GaussianNB()  
        # KNeighborsClassifier(),
        # DecisionTreeClassifier(),
        # RandomForestClassifier(),
        MLPClassifier()      
        ]
    for clf in  classifiers:
        model = clf.fit(X_train, y_train)
        # print("classifier", model)
        # print ("Accuracy on Train Set")
        # print (model.score(X_train, y_train))
        # print ("Accuracy on Test Set")
        # print (model.score(X_test, y_test))        
        # print ("Report")
        # print (classification_report(y_test,model.predict(X_test))) 
        # print ("Confusion Matrix")
        confusion_matrices[str(clf)] = confusion_matrix(y_test,model.predict(X_test))
        roc_and_auc(confusion_matrices[str(clf)])
        # print (confusion_matrix(y_test,model.predict(X_test))) 
    return confusion_matrices

def roc_and_auc(confusion_matrix_for_a_model):
    print ("Confusion Matrix")
    print (confusion_matrix_for_a_model)
    print ("Wirte the Code Here Sir")

Data = []
Output = []

#Exporting the CSV paths
#Path = 'datasets/NN/16_riboswitches.csv'
Path = 'datasets/NN/24_riboswitches.csv'

#Call function to Load Dataset
Data, Output = Create_Data(Path, Data, Output)

#Divide Dataset for training and testing
Data_train, Data_test, Output_train, Output_test = train_test_split(Data, Output, test_size=0.2, stratify=Output)

#Converting the train data into Float
Data_train, Output_train = Convert_to_Float(Data_train, Output_train)

#Converting the test data into Float
Data_test, Output_test = Convert_to_Float(Data_test, Output_test)

#Preprocessing the data
scaler = StandardScaler()
scaler.fit(Data_train)
Data_train = scaler.transform(Data_train)
Data_test = scaler.transform(Data_test)

confu = construct_models(Data_train, Data_test, Output_train, Output_test)

# Ramit Calc 
True_Positives, False_Negatives, All_Positives, False_Positives, True_Negatives, All_Negatives = roc.choose_from_confusion_matrix(confu)

# Recall = roc.Rec(True_Positives,All_Positives)
# Precision = roc.Pre(True_Positives,False_Positives)
# Accuracy = roc.Acc(True_Positives,True_Negatives,False_Positives,False_Negatives)
# Precisionf = roc.Pre(True_Positives,False_Positives,average='False')
# Recallf, tprf = roc.Rec(True_Positives,All_Positives,average='False')
# F1 = roc.F(Precisionf,Recallf)
# FPR = roc.fdr(False_Positives,All_Negatives)
# FPRf, fprf = roc.fdr(False_Positives,All_Negatives,average='False')
# roc.display_graphs(Precision, Recall, Accuracy, F1, FPR)

# print ("For ROC")
# True Positive Rate (24 classes)
# print (tprf)
# False Positive Rate (24 classes)
# print (fprf)
# print (len(tprf[0]))
# print (len(fprf[0]))

# Ramit Calc End 

# for x, y in zip(fprf, tprf):
#     print (x)
#     print (y)
#     roc_auc = metrics.auc(x, y)

# print (roc_auc)

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
