from keras.layers import Dense, Dropout, LSTM, Embedding, Activation, Lambda
from keras.engine import Input, Model, InputSpec
from keras.preprocessing.sequence import pad_sequences
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.models import model_from_json
from keras import backend as K
from sklearn import decomposition
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
# from Bio import SeqIO
# import argparse

MAXLEN = 250

def letter_to_index(letter):
    _alphabet = 'ATGCN'
    return next((i for i, _letter in enumerate(_alphabet) if _letter == letter), None)

def load_test(input_file):
    print ('Loading data...')
    df = pd.read_csv(input_file)
    df['sequence'] = df['sequence'].apply(lambda x: [int(letter_to_index(e)) for e in x])
    # df = df.reindex(np.random.permutation(df.index))
    sample = np.array(df['sequence'].values[:len(df)])
    return pad_sequences(sample, maxlen=MAXLEN)  

input_file = 'cleanData/preditionSample.csv'
# print('Predict samples...')
# model.load_weights('modelRibo.h5')
model = load_model('modelRibo250.h5')

json_file = open('modelRibo250.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

X = load_test(input_file)
y = model.predict_classes(X, verbose=0)

# show the inputs and predicted outputs
print (y)