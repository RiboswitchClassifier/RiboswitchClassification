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
plt.rcParams['figure.figsize'] = (16, 4)
sb.set_style('whitegrid')

# def save_data_visulisation(sns_plot):
#     sns_plot.savefig("classCategories.png", dpi=1000)

def count_plot(y):
    ax= sb.countplot(y,label="Count")      
    # T1, T2, T3 = y.value_counts()
    # print('Number of T1: ',T1)
    # print('Number of T2 : ',T2)
    # print('Number of T2 : ',T3)  
    fig = ax.get_figure()
    fig.savefig('classCategories.png')
    print (y.value_counts())  
    plt.show()
    # plt.savefig("classCategories.png", dpi=1000)
    # save_data_visulisation(sns_plot)

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

# input_file_test = 'cleanData/preditionSample.csv'
input_file_test = "cleanData/shuffled16riboswitches.csv"
x, y, y_for_plotting = load_test(input_file_test)
model = load_model('epochTuning/FinalmodelRibo10Epoch.h5')
y_pred = model.predict_classes(x)
# count_plot(y_for_plotting)

print (y_pred.shape)
print (y_pred)
print (y.shape)
print (y)
print ("Report")
print(classification_report(y, y_pred))
  