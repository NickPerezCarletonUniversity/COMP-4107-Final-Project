#trying to implement ensemble method
#https://datascience.stackexchange.com/questions/27169/taking-average-of-multiple-neural-networks
#mixture of experts 'with kmeans'
#https://en.wikipedia.org/wiki/Mixture_of_experts
#combining models together university of Tartu
#https://courses.cs.ut.ee/MTAT.03.277/2014_fall/uploads/Main/deep-learning-lecture-9-combining-multiple-neural-networks-to-improve-generalization-andres-viikmaa.pdf
import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

(trX, trY), (teX, teY) = tf.keras.datasets.fashion_mnist.load_data()

trX = trX.reshape(60000, 784)
teX = teX.reshape(10000, 784)
trX = np.vstack((trX, teX))


print(trX.shape)

folds = 10
size_of_fold = int(len(trX)/folds)

answer = []
for i in range(2,10):
    print(i)
    for fold in range(0,folds):
        temp = np.vstack((trX[0:fold*size_of_fold], trX[(fold + 1)*size_of_fold:len(trX)]))
        print(temp.shape)
        kmeans = KMeans(n_clusters=i, random_state=0).fit(temp)
        np.save('kmeansclusters/' + str(i) + 'fold' + str(fold) + '.npy', kmeans.cluster_centers_)
        np.save('kmeanslabels/' + str(i) + 'fold' + str(fold) + '.npy', kmeans.labels_)
