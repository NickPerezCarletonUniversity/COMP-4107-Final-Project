import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import time

(trX, trY), (teX, teY) = tf.keras.datasets.fashion_mnist.load_data()

trX = trX.reshape(60000, 784)
teX = teX.reshape(10000, 784)
trX = np.vstack((trX, teX))


print(trX.shape)

folds = 10
size_of_fold = int(len(trX)/folds)

average_time_to_train = []
times_to_train = []

begin = 2
end = 11

for i in range(begin, end):
    times_to_train.append([])

for i in range(begin, end):
    print(i)
    for fold in range(0,folds):
        temp = np.vstack((trX[0:fold*size_of_fold], trX[(fold + 1)*size_of_fold:len(trX)]))
        print(temp.shape)
        time_start = time.process_time()
        kmeans = KMeans(n_clusters=i, random_state=0).fit(temp)
        time_stop = time.process_time()
        times_to_train[i-begin].append(time_stop - time_start)
        np.save('kmeansclusters/' + str(i) + 'fold' + str(fold) + '.npy', kmeans.cluster_centers_)

times_to_train = np.asarray(times_to_train)

print(times_to_train)

times_to_train = np.asarray(times_to_train)

np.save('kmeans_computation_time_per_fold.npy', times_to_train)

        
    