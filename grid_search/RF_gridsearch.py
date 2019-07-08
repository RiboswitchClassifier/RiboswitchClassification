from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
import pandas as pd
import csv
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import RandomForestClassifier
"""
def Create_Data(Path, Data, Output):
    with open(Path) as csvfile:
        Data_Path = list(csv.DictReader(csvfile))
        for x in Data_Path:
                #Creating the feature vector of mono and di nucleotides
                Data.append([x["A"], x["T"], x["G"], x["C"],x["AA"], x["AC"], x["AG"], x["AT"],x["CA"], x["CC"], x["CG"], x["CT"],x["GA"], x["GC"], x["GG"], x["GT"],x["TA"], x["TC"], x["TG"], x["TT"]])
                Output.append(x["Type"])
        return Data, Output


def Convert_to_Float(Data, Output):
    for i in range(len(Data)):
        for j in range(20):
            Data[i][j]=float(Data[i][j])
        Output[i]= int(Output[i])
    return Data, Output
Data_train = []
Output_train = []
Data_test = []
Output_test = []
bin_output = []

Path = 'processed_datasets/final_32train.csv'

#Call function to Load Dataset
Data_train, Output_train = Create_Data(Path, Data_train, Output_train)

#Converting the train data into Float
Data_train, Output_train = Convert_to_Float(Data_train, Output_train)

Path = 'processed_datasets/final_32test.csv'

#Call function to Load Dataset
Data_test, Output_test = Create_Data(Path, Data_test, Output_test)

#Converting the train data into Float
Data_test, Output_test = Convert_to_Float(Data_test, Output_test)

unique_classes = list(set(Output_test))
unique_classes.sort()

bin_output = label_binarize(Output_test, classes=unique_classes)


#Preprocessing the data
scaler = StandardScaler()
scaler.fit(Data_train)
Data_train = scaler.transform(Data_train)
Data_test = scaler.transform(Data_test)

model = MLPClassifier().fit(Data_train, Output_train)
print("classifier", model)
print ("Accuracy on Train Set")
print (model.score(Data_train, Output_train))
print ("MLP Classifier")
print ("Accuracy on Test Set")
print (model.score(Data_test, Output_test))
print ("Report")
print (classification_report(Output_test,model.predict(Data_test)))

mlp = MLPClassifier(max_iter=100)
parameter_space = {
    #'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    #'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    #'learning_rate': ['constant','adaptive'],
}
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=2)
clf.fit(Data_train, Output_train)
# Best parameter set
print('Best parameters found:\n', clf.best_params_)

# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

y_true, y_pred = Output_test , clf.predict(Output_train)

print('Results on the test set:')
print(classification_report(y_true, y_pred))
"""
data = pd.read_csv('processed_datasets/final_32classes.csv')
# Separate out the x_data and y_data.
x_data = data.loc[:, data.columns != "Type"]
x_data = x_data.loc[:,x_data.columns != "Sequence"]

y_data = data.loc[:, "Type"]
random_state = 100


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.5, random_state=100,stratify=y_data)
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



rf = RandomForestClassifier(n_estimators=3000,max_depth=100,n_jobs=-1)
rf.fit(x_train, y_train)

y_pred_train = rf.predict(x_train)
y_pred_test = rf.predict(x_test)

print("classifier", rf)
print ("Accuracy on Train Set")
print (rf.score(x_train, y_train))
print ("MLP Classifier")
print ("Accuracy on Test Set")
print (rf.score(x_test, y_test))
print ("Report")
print (classification_report(y_test,rf.predict(x_test)))

param_grid = {
    'n_estimators': [1,1000,2000,3000],
    'max_depth': [50,60,70,80,90,100]

}

#,2000
#,70

grid_search = GridSearchCV(rf, param_grid=param_grid,n_jobs=-1,cv=10)

grid_search.fit(x_train,y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
