import tensorflow as tf
import theano
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

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
    df['sequence'] = df['sequence'].apply(lambda x: [int(letter_to_index(e)) for e in x])
    # df = df.reindex(np.random.permutation(df.index))
    sample = np.array(df['sequence'].values[:len(df)])
    return pad_sequences(sample, maxlen=MAXLEN) 

# Path to Datasets to be tested
# input_file_test = "datasets/RNN/predition_sample.csv"
# input_file_test = "datasets/RNN/16_riboswitches_shuffled.csv"
input_file_test = "datasets/RNN/24_riboswitches_shuffled.csv"

X_T = load_test(input_file_test)

# Load Models
# model = load_model('models/RNN/rnn_16_model.h5')
model = load_model('models/RNN/rnn_24_model.h5')

Y_T = model.predict_classes(X_T, verbose=0) 
print ("Predicted Outcomes")
print (Y_T)   