from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
import pandas as pd
import csv
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.tree import DecisionTreeClassifier

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

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

y_pred_train = dt.predict(x_train)
y_pred_test = dt.predict(x_test)

print("classifier", dt)
print ("Accuracy on Train Set")
print (dt.score(x_train, y_train))
print ("MLP Classifier")
print ("Accuracy on Test Set")
print (dt.score(x_test, y_test))
print ("Report")
print (classification_report(y_test,dt.predict(x_test)))

param_grid = {
    'max_features': ['auto', 'sqrt', 'log2',None],
        'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15],
          'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11],
         'random_state':[123,345,None],
         'max_depth':[5,10,15,20,25,None]

}

#,2000
#,70

grid_search = GridSearchCV(dt, param_grid=param_grid,n_jobs=-1,cv=10)

grid_search.fit(x_train,y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
