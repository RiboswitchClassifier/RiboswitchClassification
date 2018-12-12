import tensorflow as tf
import theano
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from sklearn.metrics import classification_report
import seaborn as sb

MAXLEN = 250 # cuts text after number of these characters in pad_sequences

def letter_to_index(letter):
    _alphabet = 'ATGCN' # DRS
    # _alphabet = 'ATGCNDRS' # DRS
    if letter not in _alphabet:
        print ("LOl")
        print (letter)
    return next((i for i, _letter in enumerate(_alphabet) if _letter == letter), None)

def load_test(input_file):
    print ('Loading data...')
    df = pd.read_csv(input_file)
    y_for_plotting = df['target']
    df['sequence'] = df['sequence'].apply(lambda x: [int(letter_to_index(e)) for e in x])
    # df = df.reindex(np.random.permutation(df.index))
    x = np.array(df['sequence'].values[:len(df)])
    y = np.array(df['target'].values[:len(df)])
    return pad_sequences(x, maxlen=MAXLEN), y, y_for_plotting 

# Path to Datasets to be tested
# input_file_test = "datasets/RNN/predition_sample.csv"
# input_file_test = "datasets/RNN/16_riboswitches_shuffled.csv"
input_file_test = "datasets/RNN/24_riboswitches_shuffled.csv"

x, y, y_for_plotting = load_test(input_file_test)

# Load Models
# model = load_model('models/RNN/rnn_16_model.h5')
model = load_model('models/RNN/rnn_24_model.h5')
y_pred = model.predict_classes(x)

print (y_pred.shape)
print (y_pred)
print (y.shape)
print (y)
print ("Report")
print(classification_report(y, y_pred))
  