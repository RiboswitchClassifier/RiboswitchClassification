from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
import pandas as pd
import csv
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

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

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

y_pred_train = knn.predict(x_train)
y_pred_test = knn.predict(x_test)

print("classifier", knn)
print ("Accuracy on Train Set")
print (knn.score(x_train, y_train))
print ("MLP Classifier")
print ("Accuracy on Test Set")
print (knn.score(x_test, y_test))
print ("Report")
print (classification_report(y_test,knn.predict(x_test)))

param_grid = {
    'n_neighbors':[5,6,7,8,9,10],
         'leaf_size':[1,2,3,5],
            'weights':['uniform', 'distance'],
            'algorithm':['auto', 'ball_tree','kd_tree','brute'],
        #  'n_jobs':[-1]

}

#,2000
#,70

grid_search = GridSearchCV(knn, param_grid=param_grid,n_jobs=-1,cv=2)

grid_search.fit(x_train,y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid_search.cv_results_)
