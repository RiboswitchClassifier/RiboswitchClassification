from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
import pandas as pd
import csv
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv('processed_datasets/final_32classes.csv')
# Separate out the x_data and y_data.
x_data = data.loc[:, data.columns != "Type"]
x_data = x_data.loc[:,x_data.columns != "Sequence"]

y_data = data.loc[:, "Type"]
random_state = 100


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.7, random_state=100,stratify=y_data)
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



rf = RandomForestClassifier()
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
