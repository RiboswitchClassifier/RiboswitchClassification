import tensorflow as tf
import theano
import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use('pdf')
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
from sklearn.utils import shuffle
from sklearn.metrics import classification_report,confusion_matrix
import os
import pydot
from keras.models import load_model
import aucRoc
import graphviz
import functools

# Hyperparameters and Parameters
EPOCHS = 25 #  an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation.
BATCH_SIZE = 128 # a set of N samples. The samples in a batch are processed` independently, in parallel. If training, a batch results in only one update to the model.
ALLOWED_ALPHABETS = 'ATGCN' # Allowed Charecters 
INPUT_DIM = len(ALLOWED_ALPHABETS) # a vocabulary of 5 words in case of genome sequence 'ATGCN'
CLASSES = 24 # Number of Classes to Classify -> Change this to 16 when needed 
OUTPUT_DIM = 50 # Embedding output of Layer 1
RNN_HIDDEN_DIM = 62 # Hidden Layers
DROPOUT_RATIO = 0.2 # proportion of neurones not used for training
MAXLEN = 250 # cuts text after number of these characters in pad_sequences
VALIDATION_SPLIT = 0.1

# Create Directory for Checkpoints
checkpoint_dir ='epoch_tuning/RNN/dev/24_checkpoints'
os.path.exists(checkpoint_dir)

# Path to save and load Model 
model_file_h5 = "models/rnn_24_model.h5"

# Path to Dataset
input_file_train = 'processed_datasets/final_train.csv'
input_file_test  = 'processed_datasets/final_test.csv'

# Convert letters to numbers 
def letter_to_index(letter):
    if letter not in ALLOWED_ALPHABETS:
        print ("Letter not present")
        print (letter)
    return next((i for i, _letter in enumerate(ALLOWED_ALPHABETS) if _letter == letter), None)

# Charecter mapping to acheive ATGCN
def charecter_maping(x):
    repls = {'R' : 'G', 'Y' : 'T', 'M' : 'A', 'K' : 'G', 'S' : 'G', 'W' : 'A', 'H' : 'A', 'B' : 'G', 'V' : 'G', 'D' : 'G'}
    x = functools.reduce(lambda a, kv: a.replace(*kv), repls.items(), x)
    return x

# Load Data to be used for training and validation
def load_data(input_file, flag ,test_split = 0.0, maxlen = MAXLEN):
    df = pd.read_csv(input_file)
    df['Sequence'] = df['Sequence'].apply(charecter_maping)
    df['Sequence'] = df['Sequence'].apply(lambda x: [int(letter_to_index(e)) for e in x])
    # df = df.reindex(np.random.permutation(df.index))
    X = np.array(df['Sequence'].values)
    Y = np.array(df['Type'].values)
    if flag:
        global CLASSES
        number_of_classes = np.unique(Y)
        CLASSES = len(number_of_classes)
        print (CLASSES)
    return pad_sequences(X, maxlen=maxlen), Y


# Create the RNN 
def create_lstm(input_length, rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO):
    model = Sequential()
    model.add(Embedding(input_dim = INPUT_DIM, output_dim = output_dim, input_length = input_length, name='embedding_layer'))
    model.add(Bidirectional(LSTM(rnn_hidden_dim, return_sequences=True)))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(rnn_hidden_dim)))
    model.add(Dropout(dropout))
    model.add(Dense(CLASSES, activation='softmax'))
    model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy']) # binary_crossentropy # categorical_crossentropy
    return model

# Train RNN
def train_model_and_save(X_train, y_train, model):
    # Save Checkpoint
    filepath= checkpoint_dir + "/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, class_weight=class_weights,
        epochs=EPOCHS, callbacks=callbacks_list, validation_split = VALIDATION_SPLIT, verbose = 1, shuffle=True)
    model.save(model_file_h5) 
    print("Saved model to disk")
    return model

# Classification Report
def generate_classification_report(model_loaded, X_test, y_test):
    print (classification_report(y_test,model_loaded.predict_classes(X_test))) 

# Predict Classes, Probabilities, Call AucRoc Function
def generate_auc_roc(X_test, y_test):
    model_loaded = load_model(model_file_h5)
    generate_classification_report(model_loaded, X_test, y_test)
    predicted_classes = model_loaded.predict_classes(X_test)
    print ("Predicted Classes")
    print (predicted_classes)
    y_score = model_loaded.predict_proba(X_test)
    print ("Predicted Probabilities") 
    print (y_score)
    aucRoc.calculate_roc(y_test, y_score, "RnnClassifierModel")

if __name__ == '__main__':
    # Load Training Datasets
    X_train, y_train = load_data(input_file_train,True) 
    # Create Model Structure
    model = create_lstm(len(X_train[0])) 
    model.summary()
    # Load Test Datasets   
    X_test, y_test = load_data(input_file_test, False) 
    # Train Model and Save it
    model = train_model_and_save(X_train, y_train, model)
    # Generate Auc and Roc Curve
    generate_auc_roc(X_test, y_test, CLASSES)



