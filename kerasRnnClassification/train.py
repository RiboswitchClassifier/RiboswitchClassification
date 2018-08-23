import tensorflow as tf
import theano
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, LSTM, Embedding, Activation, Lambda, Bidirectional
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

EPCOHS = 10 #  an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation.
BATCH_SIZE = 500 # a set of N samples. The samples in a batch are processed` independently, in parallel. If training, a batch results in only one update to the model.
INPUT_DIM = 5 # a vocabulary of 4 words in case of fnn sequence (ATCG)
CLASSES = 16
OUTPUT_DIM = 50 # Embedding output
RNN_HIDDEN_DIM = 62
DROPOUT_RATIO = 0.2 # proportion of neurones not used for training
MAXLEN = 100 # cuts text after number of these characters in pad_sequences

# checkpoint_dir ='checkpoints'
checkpoint_dir ='newCheckpoints'
os.path.exists(checkpoint_dir)

# input_file = 'cami_all_150.csv'
# input_file = 'splice_new.csv'
input_file = 'cleanData/16riboswitches.csv'

def letter_to_index(letter):
    _alphabet = 'ATGCN' # DRS
    # _alphabet = 'ATGCNDRS' # DRS
    if letter not in _alphabet:
        print ("LOl")
        print (letter)
    return next((i for i, _letter in enumerate(_alphabet) if _letter == letter), None)

def encode(data):
    print('Shape of data (BEFORE encode): %s' % str(data.shape))
    encoded = to_categorical(data)
    print('Shape of data (AFTER  encode): %s\n' % str(encoded.shape))
    return encoded

def load_data(test_split = 0.1, maxlen = MAXLEN):
    print ('Loading data...')
    df = pd.read_csv(input_file)
    df['sequence'] = df['sequence'].apply(lambda x: [int(letter_to_index(e)) for e in x])
    df = df.reindex(np.random.permutation(df.index))
    train_size = int(len(df) * (1 - test_split))
    X_train = np.array(df['sequence'].values[:train_size])
    print (X_train)
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

def create_lstm(input_length, rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO):
    model = Sequential()
    model.add(Embedding(input_dim = INPUT_DIM, output_dim = output_dim, input_length = input_length, name='embedding_layer'))
    model.add(Bidirectional(LSTM(rnn_hidden_dim, return_sequences=True)))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(rnn_hidden_dim)))
    model.add(Dropout(dropout))
    model.add(Dense(CLASSES, activation='softmax'))

    model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy']) # binary_crossentropy
    # # For a multi-class classification problem
    # model.compile(optimizer='rmsprop',
    #             loss='categorical_crossentropy',
    #             metrics=['accuracy'])

    # # For a binary classification problem
    # model.compile(optimizer='rmsprop',
    #             loss='binary_crossentropy',
    #             metrics=['accuracy'])   

    return model

def create_plots(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy.png')
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')
    plt.clf()

if __name__ == '__main__':
    # train
    X_train, y_train, X_test, y_test = load_data()    
    model = create_lstm(len(X_train[0])) 
    model.summary()

    # save checkpoint
    filepath= checkpoint_dir + "/weightsRibo-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    print ('Fitting model...')
    print (np.unique(y_train))
    class_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    print(class_weight)
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, class_weight=class_weight,
        epochs=EPCOHS, callbacks=callbacks_list, validation_split = 0.1, verbose = 1)
    # history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, class_weight="auto", epochs=EPCOHS, callbacks=callbacks_list, validation_split = 0.1, verbose = 1)    

    # serialize model to JSON
    model_json = model.to_json()
    with open("modelRibo.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("modelRibo.h5")
    print("Saved model to disk")
    
    # create_plots(history)
    # plot_model(model, to_file='modelRibo.png')

    # validate model on unseen data
    score, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    print('Validation score:', score)
    print('Validation accuracy:', acc)
