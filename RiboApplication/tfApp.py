import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Train Accuracy:  0.268499
# Test Accuracy:  0.274108
# MSE: 1.8567

# Reading the dataset
def read_dataset():
    df = pd.read_csv("datasets/NN/16_riboswitches.csv")
    print(len(df.columns))
    X = df[df.columns[2:22]].values
    y = df[df.columns[1]]

    # Encode the dependent variable
    Y = one_hot_encode(y)
    print(X.shape)
    return (X, Y)

# Define the encoder function.
def one_hot_encode(labels):
    n_labels = len(labels)
    print (n_labels)
    n_unique_labels = len(np.unique(labels))
    print (n_unique_labels)
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


# Read the dataset
X, Y = read_dataset()

# Shuffle the dataset to mix up the rows.
X, Y = shuffle(X, Y, random_state=1)

# Convert the dataset into train and test part
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.20)

# Inpect the shape of the training and testing.
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

# Define the important parameters and variable to work with the tensors
learning_rate = 0.01
training_epochs = 250
cost_history = np.empty(shape=[1], dtype=float)
n_dim = X.shape[1]
print("n_dim", n_dim)
n_class = 16
model_path = "models/tf_16_model"

# Define the number of hidden layers and number of neurons for each layer
n_hidden_1 = 16
n_hidden_2 = 5
n_hidden_3 = 5
n_hidden_4 = 5
n_hidden_5 = 5
n_hidden_6 = 5
n_hidden_7 = 5
n_hidden_8 = 5
n_hidden_9 = 5
n_hidden_10 = 5
n_hidden_11 = 5

x = tf.placeholder(tf.float32, [None, n_dim])
W = tf.Variable(tf.zeros([n_dim, n_class]))
b = tf.Variable(tf.zeros([n_class]))
y_ = tf.placeholder(tf.float32, [None, n_class])


# Define the model
def multilayer_perceptron(x, weights, biases):

    # Hidden layer with RELU activationsd
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden layer with sigmoid activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Hidden layer with sigmoid activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)

    # Hidden layer with RELU activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    # Hidden layer with RELU activationsd
    layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
    layer_5 = tf.nn.relu(layer_5)
    
    # Hidden layer with sigmoid activation
    layer_6 = tf.add(tf.matmul(layer_5, weights['h6']), biases['b6'])
    layer_6 = tf.nn.relu(layer_6)
    
    # Hidden layer with sigmoid activation
    layer_7 = tf.add(tf.matmul(layer_6, weights['h7']), biases['b7'])
    layer_7 = tf.nn.relu(layer_7)
    
    # Hidden layer with RELU activation
    layer_8 = tf.add(tf.matmul(layer_7, weights['h8']), biases['b8'])
    layer_8 = tf.nn.relu(layer_8)
    
    # Hidden layer with sigmoid activation
    layer_9 = tf.add(tf.matmul(layer_8, weights['h9']), biases['b9'])
    layer_9 = tf.nn.relu(layer_9)
    
    # Hidden layer with sigmoid activation
    layer_10 = tf.add(tf.matmul(layer_9, weights['h10']), biases['b10'])
    layer_10 = tf.nn.relu(layer_10)
    
    # Hidden layer with RELU activation
    layer_11 = tf.add(tf.matmul(layer_10, weights['h11']), biases['b11'])
    layer_11 = tf.nn.relu(layer_11)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_11, weights['out']) + biases['out']
    return out_layer


# Define the weights and the biases for each layer

weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
    'h5': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_5])),
    'h6': tf.Variable(tf.truncated_normal([n_hidden_5, n_hidden_6])),
    'h7': tf.Variable(tf.truncated_normal([n_hidden_6, n_hidden_7])),
    'h8': tf.Variable(tf.truncated_normal([n_hidden_7, n_hidden_8])),
    'h9': tf.Variable(tf.truncated_normal([n_hidden_8, n_hidden_9])),
    'h10': tf.Variable(tf.truncated_normal([n_hidden_9, n_hidden_10])),
    'h11': tf.Variable(tf.truncated_normal([n_hidden_10, n_hidden_11])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_class]))
}

biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
    'b5': tf.Variable(tf.truncated_normal([n_hidden_5])),
    'b6': tf.Variable(tf.truncated_normal([n_hidden_6])),
    'b7': tf.Variable(tf.truncated_normal([n_hidden_7])),
    'b8': tf.Variable(tf.truncated_normal([n_hidden_8])),
    'b9': tf.Variable(tf.truncated_normal([n_hidden_9])),
    'b10': tf.Variable(tf.truncated_normal([n_hidden_10])),
    'b11': tf.Variable(tf.truncated_normal([n_hidden_11])),
    'out': tf.Variable(tf.truncated_normal([n_class]))
}

# Initialize all the variables

init = tf.global_variables_initializer()

saver = tf.train.Saver()

# Call your model defined
y = multilayer_perceptron(x, weights, biases)

# Define the cost function and optimizer
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

sess = tf.Session()
sess.run(init)

# Calculate the cost and the accuracy for each epoch

mse_history = []
accuracy_history = []

for epoch in range(training_epochs):
    sess.run(training_step, feed_dict={x: train_x, y_: train_y})
    cost = sess.run(cost_function, feed_dict={x: train_x, y_: train_y})
    cost_history = np.append(cost_history, cost)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print("Accuracy: ", (sess.run(accuracy, feed_dict={x: test_x, y_: test_y})))
    pred_y = sess.run(y, feed_dict={x: test_x})
    mse = tf.reduce_mean(tf.square(pred_y - test_y))
    mse_ = sess.run(mse)
    mse_history.append(mse_)
    accuracy = (sess.run(accuracy, feed_dict={x: train_x, y_: train_y}))
    accuracy_history.append(accuracy)

    print('epoch : ', epoch, ' - ', 'cost: ', cost, " - MSE: ", mse_, "- Train Accuracy: ", accuracy)

# save_path = saver.save(sess, model_path)
# print("Model saved in file: %s" % save_path)

# #Plot Accuracy Graph
# plt.plot(accuracy_history)
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.show()
#
# Print the final accuracy

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Test Accuracy: ", (sess.run(accuracy, feed_dict={x: test_x, y_: test_y})))

# Print the final mean square error

pred_y = sess.run(y, feed_dict={x: test_x})
mse = tf.reduce_mean(tf.square(pred_y - test_y))
print("MSE: %.4f" % sess.run(mse))
