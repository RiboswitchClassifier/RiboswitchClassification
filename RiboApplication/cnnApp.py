import tensorflow as tf
# import theano
import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use('pdf')
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, LSTM, Embedding, Activation, Lambda, Bidirectional, Flatten
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
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
from sklearn.utils import shuffle
import os
import pydot
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from collections import Counter
import roc
from keras.models import load_model
import aucRoc

# 16 and 24 classes gave similar metrics
# Train Accuracy : 0.98
# Validation Accuracy : 0.98
# Test Accuracy : 0.97
# F1 Score : 0.98

# Hyperparameters and Parameters
EPOCHS = 20 #  an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation.
BATCH_SIZE = 128 # a set of N samples. The samples in a batch are processed` independently, in parallel. If training, a batch results in only one update to the model.
INPUT_DIM = 5 # a vocabulary of 5 words in case of genome sequence 'ATGCN'
CLASSES = 24 # Number of Classes to Classify -> Change this to 16 when needed 
OUTPUT_DIM = 50 # Embedding output of Layer 1
RNN_HIDDEN_DIM = 62 # Hidden Layers 
DROPOUT_RATIO = 0.2 # proportion of neurones not used for training
MAXLEN = 250 # cuts text after number of these characters in pad_sequences

# Create Directory for Checkpoints
# checkpoint_dir ='epoch_tuning/RNN/16_checkpoints'
# checkpoint_dir ='epoch_tuning/RNN/24_checkpoints'
checkpoint_dir ='epoch_tuning/RNN/dev/24_checkpoints'
os.path.exists(checkpoint_dir)

# Path to save and load Model
# model_file_json = "models/RNN/rnn_16_model.json"
# model_file_h5 = "models/RNN/rnn_16_model.h5" 
# model_file_json = "models/RNN/rnn_24_model.json"
# model_file_h5 = "models/RNN/rnn_24_model.h5" 
model_file_json = "models/RNN/dev/rnn_24_model.json"
model_file_h5 = "models/cnn_24_model.h5"

# Path to Dataset
# input_file = 'datasets/RNN/16_riboswitches_shuffled.csv'
# input_file = 'datasets/RNN/24_riboswitches_shuffled.csv'

# Just to check if things are working properly
input_file_test = 'datasets/RNN/predition_sample.csv'

# Convert letters to numbers 
def letter_to_index(letter):
    _alphabet = 'ATGCN' # DRS
    # _alphabet = 'ATGCNDRS' # DRS
    if letter not in _alphabet:
        print ("LOl")
        print (letter)
    return next((i for i, _letter in enumerate(_alphabet) if _letter == letter), None)

# Load Data to be tested (After the model has been built)
def load_test(input_file):
    print ('Loading data...')
    df = pd.read_csv(input_file)
    df['Sequence'] = df['Sequence'].apply(lambda x: [int(letter_to_index(e)) for e in x])
    # df = df.reindex(np.random.permutation(df.index))
    sample = np.array(df['Sequence'].values[:len(df)])
    return pad_sequences(sample, maxlen=MAXLEN) 

# Load Data to be used for training and validation
def load_data(input_file, test_split = 0.0, maxlen = MAXLEN):
    onehot_encoder = OneHotEncoder(sparse=False)
    print ('Loading data...')
    df = pd.read_csv(input_file)
    df['Sequence'] = df['Sequence'].apply(lambda x: [int(letter_to_index(e)) for e in x])
    df = df.reindex(np.random.permutation(df.index))
    # train_size = int(len(df) * (1 - test_split))
    # X_train = np.array(df['Sequence'].values[:train_size])
    # # print (X_train)
    # y_train = np.array(df['Type'].values[:train_size])
    # # y_train = encode(y_train)
    # X_test = np.array(df['Sequence'].values[train_size:])
    # y_test = np.array(df['Type'].values[train_size:])
    # # y_test = encode(y_test)
    X = np.array(df['Sequence'].values)
    Y = np.array(df['Type'].values)
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, stratify=Y) 
    # print ("Test Set")
    # print (y_test)
    # print (set(y_test))
    # print (Counter(y_test))
    # print (Counter(Y))
    # print('Average train sequence length: {}'.format(np.mean(list(map(len, X_train)), dtype=int)))
    # print('Average test sequence length: {}'.format(np.mean(list(map(len, X_test)), dtype=int)))
    # print (X_train.shape)
    # print (y_train.shape)
    # print (X_test.shape)
    # print (y_test.shape)   
    # return pad_sequences(X_train, maxlen=maxlen), y_train, pad_sequences(X_test, maxlen=maxlen), y_test
    return pad_sequences(X, maxlen=maxlen), Y

# Create the RNN 
def create_lstm(input_length, X_train ,rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO):
    m, n =  X_train.shape
    print (m)
    print (input_length)
    model = Sequential()
    model.add(Conv1D(filters = 10, kernel_size = 3, input_shape=(input_length, 1)))
    model.add(Conv1D(filters = 10, kernel_size = 3, activation='relu'))
    model.add(MaxPooling1D(3))
    # model.add(Conv1D(filters = 10, kernel_size = 3, activation='relu'))
    # model.add(Conv1D(filters = 10, kernel_size = 3, activation='relu'))
    # model.add(GlobalAveragePooling1D())    
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(CLASSES, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])    
    # model.add(Embedding(input_dim = INPUT_DIM, output_dim = output_dim, input_length = input_length, name='embedding_layer'))
    # model.add(Bidirectional(LSTM(rnn_hidden_dim, return_sequences=True)))
    # model.add(Dropout(dropout))
    # model.add(Bidirectional(LSTM(rnn_hidden_dim)))
    # model.add(Dropout(dropout))
    # model.add(Dense(CLASSES, activation='softmax'))
    # model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy']) # binary_crossentropy # categorical_crossentropy

    # # For a multi-class classification problem
    # model.compile(optimizer='rmsprop',
    #             loss='categorical_crossentropy',
    #             metrics=['accuracy'])

    # # For a binary classification problem
    # model.compile(optimizer='rmsprop',
    #             loss='binary_crossentropy',
    #             metrics=['accuracy'])   

    return model

# This function is not used
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
    # Load Datasets and Create RNN Schema
    # input_file = 'processed_datasets/24_riboswitches_final.csv'
    # input_file = 'processed_datasets/24_riboswitches_final_train.csv'
    # X_train, y_train = load_data(input_file)  
    input_file = 'processed_datasets/24_riboswitches_final_test.csv'
    X_test, y_test = load_data(input_file) 
    




    

    # model = create_lstm(len(X_train[0]), X_train) 
    # model.summary()

    # # # Save Checkpoint
    # # filepath= checkpoint_dir + "/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    # # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    # # callbacks_list = [checkpoint]

    # print ('Fitting model...')
    # print (np.unique(y_train))
    # class_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train) # y_ints = [y.argmax() for y in y_train]
    # print ("Class Weights")
    # print(class_weight)
    # X_train = np.expand_dims(X_train, axis=2)
    # print (X_train.shape)
    # history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, class_weight=class_weight,
    #     epochs=EPOCHS, validation_split = 0.1, verbose = 1, shuffle=True)
    # # history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, class_weight="auto", epochs=EPOCHS, callbacks=callbacks_list, validation_split = 0.1, verbose = 1)    
    # # plot_model(model, to_file='modelCNN.png')
    # # # serialize model to JSON file format
    # # model_json = model.to_json()
    # # with open(model_file_json, "w") as json_file:
    # #     json_file.write(model_json)

    # # serialize weights to HDF5 file format
    # # model.save_weights(model_file_h5)
    # model.save(model_file_h5) 
    # print("Saved model to disk")
    
    # # # create_plots(history)
    # # # plot_model(model, to_file='modelRibo.png')





    # Validate the model
    model_file_h5 = "models/cnn_24_model.h5"
    X_test = np.expand_dims(X_test, axis=2)
    model_loaded = load_model(model_file_h5)
    print ("GG")
    loss, acc = model_loaded.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    print('Test Loss:', loss)
    print('Test Accuracy:', acc)

    # # Did this to check if the right results were actually coming
    # X_T = load_test(input_file_test)
    # X_T = np.expand_dims(X_T, axis=2)
    # Y_T = model.predict_classes(X_T, verbose=0)

    # # show the inputs and predicted outputs
    # print ("Predicted Outcomes")
    # print (Y_T) 

    # X_test = np.expand_dims(X_test, axis=2)

    # confusion_matrices = {}
    # confusion_matrices["cnn"] = confusion_matrix(y_test,model_loaded.predict_classes(X_test)) 
    # print (confusion_matrices["cnn"])
    # print (y_test)
    # print (model_loaded.predict_classes(X_test))
    print ("Classification Report")
    print (classification_report(y_test,model_loaded.predict_classes(X_test))) 
    print ("Predicted Score")
    y_score = model_loaded.predict_proba(X_test) 
    print (y_score)

    aucRoc.calculate_roc(y_test, y_score, "CnnClassifierModel")

    # True_Positives, False_Negatives, All_Positives, False_Positives, True_Negatives, All_Negatives = roc.choose_from_confusion_matrix(confusion_matrices)

    # Recall = roc.Rec(True_Positives,All_Positives)
    # Precision = roc.Pre(True_Positives,False_Positives)
    # Accuracy = roc.Acc(True_Positives,True_Negatives,False_Positives,False_Negatives)
    # Precisionf = roc.Pre(True_Positives,False_Positives,average='False')
    # Recallf = roc.Rec(True_Positives,All_Positives,average='False')
    # F1 = roc.F(Precisionf,Recallf)
    # print (F1)
    # FPR = roc.fdr(False_Positives,All_Negatives)
    # roc.display_graphs(Precision, Recall, Accuracy, F1, FPR)  
    print ("hi")  
   # tpr-> and fpr-> 1 - specifty 


