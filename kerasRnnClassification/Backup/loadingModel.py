import tensorflow as tf
import theano
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, LSTM, Embedding, Activation, Lambda, Bidirectional
from sklearn.preprocessing import OneHotEncoder
from keras.engine import Input, Model, InputSpec
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.utils import class_weight
from keras import backend as K
from keras.preprocessing import sequence
from keras.models import model_from_json
from keras.utils import to_categorical
import os
import pydot
import graphviz

EPOCHS = 10 #  an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation.
BATCH_SIZE = 128 # a set of N samples. The samples in a batch are processed` independently, in parallel. If training, a batch results in only one update to the model.
INPUT_DIM = 5 # a vocabulary of 4 words in case of fnn sequence (ATCG)
CLASSES = 16
OUTPUT_DIM = 50 # Embedding output
RNN_HIDDEN_DIM = 62
DROPOUT_RATIO = 0.2 # proportion of neurones not used for training
MAXLEN = 250 # cuts text after number of these characters in pad_sequences

def load_data(test_split = 0.1, maxlen = MAXLEN):
    input_file = 'cleanData/shuffled16riboswitches.csv'
    onehot_encoder = OneHotEncoder(sparse=False)
    print ('Loading data...')
    df = pd.read_csv(input_file)
    df['sequence'] = df['sequence'].apply(lambda x: [int(letter_to_index(e)) for e in x])
    df = df.reindex(np.random.permutation(df.index))
    train_size = int(len(df) * (1 - test_split))
    X_train = np.array(df['sequence'].values[:train_size])
    # print (X_train)
    y_train = np.array(df['target'].values[:train_size])
    # y_train = encode(y_train)
    X_test = np.array(df['sequence'].values[train_size:])
    y_test = np.array(df['target'].values[train_size:])
    # y_test = encode(y_test)
    print('Average train sequence length: {}'.format(np.mean(list(map(len, X_train)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, X_test)), dtype=int)))
    print (X_train.shape)
    print (y_train.shape)
    print (X_test.shape)
    print (y_test.shape)    
    return pad_sequences(X_train, maxlen=maxlen), y_train, pad_sequences(X_test, maxlen=maxlen), y_test

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

def create_model(input_length, rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO):
    model = Sequential()
    model.add(Embedding(input_dim = INPUT_DIM, output_dim = output_dim, input_length = input_length, name='embedding_layer'))
    model.add(Bidirectional(LSTM(rnn_hidden_dim, return_sequences=True)))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(rnn_hidden_dim)))
    model.add(Dropout(dropout))
    model.add(Dense(CLASSES, activation='softmax'))
    return model

def load_trained_model(weights_path):
    X_train, y_train, X_test, y_test = load_data()  
    print("Obtaining Length")
    print (len(X_train[0]))
    model = create_model(len(X_train[0]))
    print ("Model Created")
    model.load_weights(weights_path)   
    print ("Model Weights Loaded")
    model.save('epochTuning/FinalmodelRibo10Epoch.h5')
    print ("Model Saved")
    input_file_test = 'cleanData/preditionSample.csv'
    X_T = load_test(input_file_test)
    Y_T = model.predict_classes(X_T, verbose=0) 
    print ("Predicted Outcomes")
    print (Y_T)        

load_trained_model("epochTuning/modelRibo10Epoch.h5") 

print ("Final")