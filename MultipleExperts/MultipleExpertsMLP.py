#!/usr/bin/env python
# coding: utf-8

# In[1]:


#trying to implement ensemble method
#https://datascience.stackexchange.com/questions/27169/taking-average-of-multiple-neural-networks
#mixture of experts 'with kmeans'
#https://en.wikipedia.org/wiki/Mixture_of_experts
#combining models together university of Tartu
#https://courses.cs.ut.ee/MTAT.03.277/2014_fall/uploads/Main/deep-learning-lecture-9-combining-multiple-neural-networks-to-improve-generalization-andres-viikmaa.pdf
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import keras
from keras.utils import to_categorical

(trX, trY), (teX, teY) = tf.keras.datasets.fashion_mnist.load_data()
trX = trX.reshape(60000, 784)
teX = teX.reshape(10000, 784)

trY = to_categorical(trY)
teY = to_categorical(teY)


print("x_train shape:", trX.shape, "y_train shape:", trY.shape)
print("x_test shape:", teX.shape, "y_test shape:", teY.shape)


# In[3]:


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h1, w_h2, w_o):
    h1 = tf.nn.sigmoid(tf.matmul(X, w_h1)) # this is a basic mlp, think 2 stacked logistic regressions
    h = tf.nn.sigmoid(tf.matmul(h1, w_h2))
    #return tf.matmul(h, w_o, name="insertname_here") if we need to use names and save the models
    return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us


prediction = 0

average_model_accuracy = 0

models_to_train = 5

epochs_per_model = 2

batch_size = 128

cluster_centers = np.load('kmeansclusters/' + str(models_to_train) + '.npy')
cluster_labels = np.load('kmeanslabels/' + str(models_to_train) + '.npy')

partitioned_train_data = []
partitioned_test_data = []
partitioned_train_labels = []
partitioned_test_labels = []

for i in range(0,models_to_train):
    partitioned_train_data.append([])
    partitioned_test_data.append([])
    partitioned_train_labels.append([])
    partitioned_test_labels.append([])

for i in range(0,models_to_train):
    for j in range(0,len(trX)):
        if cluster_labels[j] == i:
            partitioned_train_data[i].append(trX[j])
            partitioned_train_labels[i].append(trY[j])
    for j in range(len(trX),len(teX) + len(trX)):
        if cluster_labels[j] == i:
            partitioned_test_data[i].append(teX[j])
            partitioned_test_labels[i].append(teY[j])

    partitioned_train_data[i] = np.vstack(partitioned_train_data[i])
    partitioned_test_data[i] = np.vstack(partitioned_test_data[i])
    partitioned_train_labels[i] = np.vstack(partitioned_train_labels[i])
    partitioned_test_labels[i] = np.vstack(partitioned_test_labels[i])


#saver = tf.train.Saver()

# Launch the graph in a session
with tf.Session() as sess:
    for z in range(0,models_to_train):
        size_h1 = tf.constant(625, dtype=tf.int32)
        size_h2 = tf.constant(300, dtype=tf.int32)

        X = tf.placeholder("float", [None, 784])
        Y = tf.placeholder("float", [None, 10])

        w_h1 = init_weights([784, size_h1]) # create symbolic variables
        w_h2 = init_weights([size_h1, size_h2])
        w_o = init_weights([size_h2, 10])

        py_x = model(X, w_h1, w_h2, w_o)
        
        trX = partitioned_train_data[z]
        teX = partitioned_test_data[z]
        trY = partitioned_train_labels[z]
        teY = partitioned_test_labels[z]

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute costs
        train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
        predict_op = tf.argmax(py_x, 1)
        tf.global_variables_initializer().run()
        for i in range(epochs_per_model):
            for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size)):
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
            print(i, np.mean(np.argmax(teY, axis=1) ==
                             sess.run(predict_op, feed_dict={X: teX})))
        model_accuracy = np.sum(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={X: teX})) + model_accuracy
        
    print("accuracy: " + str(model_accuracy/10000))
        
        #saver.save(sess,"mlp/session.ckpt")


# In[ ]:




