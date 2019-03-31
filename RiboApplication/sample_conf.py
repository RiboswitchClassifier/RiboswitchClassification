from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import pickle

with open ('outfile', 'rb') as fp:
    confusion_matrix = pickle.load(fp)

print ("Confusion Matrix")
print (confusion_matrix)