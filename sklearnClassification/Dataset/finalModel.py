# Python 2.x

import csv
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

scaler = StandardScaler()

X = []
y = []

lysinePath = 'lysine_frequency.csv'
mocoPath = 'moco_frequency.csv'
purinePath = 'purine_frequency.csv'
samPath = 'sam_frequency.csv'

with open(lysinePath) as csvfile:
    lysineData = list(csv.DictReader(csvfile))

with open(mocoPath) as csvfile:
    mocoData = list(csv.DictReader(csvfile))

with open(purinePath) as csvfile:
    purineData = list(csv.DictReader(csvfile))

with open(samPath) as csvfile:
    samData = list(csv.DictReader(csvfile))

# print "Features"
# print data[0].keys()

# Lysine
for x in lysineData:
	X.append([x["A"], x["T"], x["G"], x["C"],
              x["AA"], x["AC"], x["AG"], x["AU"],
              x["CA"], x["CC"], x["CG"], x["CU"],
              x["GA"], x["GC"], x["GG"], x["GU"],
              x["UA"], x["UC"], x["UG"], x["UU"]
              ])
	y.append(x["Type"])

# Moco
for x in mocoData:
	X.append([x["A"], x["T"], x["G"], x["C"],
              x["AA"], x["AC"], x["AG"], x["AU"],
              x["CA"], x["CC"], x["CG"], x["CU"],
              x["GA"], x["GC"], x["GG"], x["GU"],
              x["UA"], x["UC"], x["UG"], x["UU"]
              ])
	y.append(x["Type"])

# Purine
for x in purineData:
	X.append([x["A"], x["T"], x["G"], x["C"],
              x["AA"], x["AC"], x["AG"], x["AU"],
              x["CA"], x["CC"], x["CG"], x["CU"],
              x["GA"], x["GC"], x["GG"], x["GU"],
              x["UA"], x["UC"], x["UG"], x["UU"]
              ])
	y.append(x["Type"])

# Sam
for x in samData:
	X.append([x["A"], x["T"], x["G"], x["C"],
              x["AA"], x["AC"], x["AG"], x["AU"],
              x["CA"], x["CC"], x["CG"], x["CU"],
              x["GA"], x["GC"], x["GG"], x["GU"],
              x["UA"], x["UC"], x["UG"], x["UU"]
              ])
	y.append(x["Type"])

# Convert to Float for calculation purposes
for i in xrange(len(X)):
    for j in xrange(20):
        X[i][j]=float(X[i][j])
    y[i]=float(y[i])

scaler.fit(X)
X = scaler.transform(X)

mlp = MLPClassifier(solver='sgd',alpha=0.0001,learning_rate='adaptive', hidden_layer_sizes=(11, 30), max_iter=300000, activation='logistic', verbose=True)
mlp.fit(X,y)
predictions_probabilities = mlp.predict_proba([[53,0,49,31,17,4,18,9,7,5,7,12,19,12,8,8,4,10,15,12]])
predictions = mlp.predict([[53,0,49,31,17,4,18,9,7,5,7,12,19,12,8,8,4,10,15,12]])
scoring = mlp.score(X, y, sample_weight=None)

# Probability of each class for the prediction input
print predictions_probabilities

# max(predictions_probabilities)
print predictions

# Score
print scoring


# print X
# print y
# #Frequency
# for x in xrange(0,10):
# 	print X[x]
#
# #Type
# for x in xrange():
# 	print y[x]
