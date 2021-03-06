#portions of this code were taken from the in class MLP.py example
import tensorflow as tf
import numpy as np
import time
import math
#import keras
#from keras.utils import to_categorical

(trX, trY), (teX, teY) = tf.keras.datasets.fashion_mnist.load_data()
trX = trX.reshape(60000, 784)
teX = teX.reshape(10000, 784)

#trY = to_categorical(trY)
#teY = to_categorical(teY)

#np.save('one_hot_train_labels.npy', trY)
#np.save('one_hot_test_labels.npy', teY)

trY = np.load('one_hot_train_labels.npy')
teY = np.load('one_hot_test_labels.npy')


print("x_train shape:", trX.shape, "y_train shape:", trY.shape)
print("x_test shape:", teX.shape, "y_test shape:", teY.shape)

total_data_set = np.vstack((trX, teX))
total_label_set = np.vstack((trY, teY))

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h1, w_h2, w_o):
    h1 = tf.nn.sigmoid(tf.matmul(X, w_h1))
    h = tf.nn.sigmoid(tf.matmul(h1, w_h2))
    return tf.matmul(h, w_o)

#taken from https://stackoverflow.com/questions/47518609/for-loop-range-and-interval-how-to-include-last-step
def myRange(start,end,step):
    i = start
    while i < end:
        yield i
        i += step
    yield end

def confidence_interval(data):
    #using 95 percent confidence interval
    data_points = len(data)
    average_accuracy = 0
    for i in range(0,data_points):
        average_accuracy = average_accuracy + data[i]/data_points

    standard_deviation = 0
    variance = 0
    for i in range(0,data_points):
        variance = variance + ((data[i] - average_accuracy) * (data[i] - average_accuracy))
    
    standard_deviation = np.sqrt(variance/data_points)
    standard_error = standard_deviation/np.sqrt(data_points)

    return [average_accuracy - 1.96*standard_error, average_accuracy, average_accuracy + 1.96*standard_error]

prediction = 0

model_accuracy = 0

models_to_train_begin = 2
models_to_train_end = 10

confidence_interval_across_experiments = []
confidence_interval_across_experiments_time = []
kfold_times = []

epochs_per_model = 3

batch_size = 128

number_of_folds = 10

multiple_experts_accuracies = []
multiple_experts_time = []

# Launch the graph in a session
with tf.Session() as sess:
    for n in range(models_to_train_begin, models_to_train_end + 1):
        models_to_train = n
        multiple_experts_accuracies = []
        multiple_experts_time = []
        for fold in range(0, number_of_folds):
            model_accuracy = 0

            cluster_centers = np.load('K-MeansClusters/' + str(models_to_train) + 'fold' + str(fold) + '.npy')

            print("fold number: " + str(fold))
            size_of_fold = int(len(total_data_set)/number_of_folds)
            trX = np.vstack((total_data_set[0:fold*size_of_fold], total_data_set[(fold + 1)*size_of_fold:len(total_data_set)]))
            teX = total_data_set[fold*size_of_fold:(fold + 1)*size_of_fold]

            trY = np.vstack((total_label_set[0:fold*size_of_fold], total_label_set[(fold + 1)*size_of_fold:len(total_label_set)]))
            teY = total_label_set[fold*size_of_fold:(fold + 1)*size_of_fold]

            print("x_train shape:", trX.shape, "y_train shape:", trY.shape)
            print("x_test shape:", teX.shape, "y_test shape:", teY.shape)

            partitioned_train_data = []
            partitioned_test_data = []
            partitioned_train_labels = []
            partitioned_test_labels = []

            for i in range(0,models_to_train):
                partitioned_train_data.append([])
                partitioned_test_data.append([])
                partitioned_train_labels.append([])
                partitioned_test_labels.append([])

            for j in range(0,len(teX)):
                closest_index = 0
                closest_distance = math.inf
                for y in range(0, len(cluster_centers)):
                    temp_distance = np.linalg.norm(cluster_centers[y] - teX[j])
                    if closest_distance > temp_distance:
                        closest_index = y
                        closest_distance = temp_distance
                partitioned_test_data[closest_index].append(teX[j])
                partitioned_test_labels[closest_index].append(teY[j])

            for j in range(0,len(trX)):
                closest_index = 0
                closest_distance = math.inf
                for y in range(0, len(cluster_centers)):
                    temp_distance = np.linalg.norm(cluster_centers[y] - trX[j])
                    if closest_distance > temp_distance:
                        closest_index = y
                        closest_distance = temp_distance
                partitioned_train_data[closest_index].append(trX[j])
                partitioned_train_labels[closest_index].append(trY[j])

            for i in range(0,models_to_train):
                partitioned_train_data[i] = np.vstack(partitioned_train_data[i])
                partitioned_test_data[i] = np.vstack(partitioned_test_data[i])
                partitioned_train_labels[i] = np.vstack(partitioned_train_labels[i])
                partitioned_test_labels[i] = np.vstack(partitioned_test_labels[i])

                print(partitioned_train_data[i].shape)
                print(partitioned_test_data[i].shape)
                print(partitioned_train_labels[i].shape)
                print(partitioned_test_labels[i].shape)

            average_model_accuracy = 0
            aggregated_model_times = 0
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
                    time_start = time.process_time()
                    for start, end in zip(myRange(0, len(trX), batch_size), myRange(batch_size, len(trX)+1, batch_size)):
                        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
                    time_stop = time.process_time()
                    aggregated_model_times = aggregated_model_times + (time_stop - time_start)
                    print(i, np.mean(np.argmax(teY, axis=1) ==
                                     sess.run(predict_op, feed_dict={X: teX})))
                model_accuracy = np.sum(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={X: teX})) + model_accuracy


            print("accuracy: " + str(model_accuracy/(len(total_data_set)/number_of_folds)))
            multiple_experts_accuracies.append(model_accuracy/(len(total_data_set)/number_of_folds))
            multiple_experts_time.append(aggregated_model_times)

        answer = confidence_interval(multiple_experts_accuracies)
        print(answer)
        confidence_interval_across_experiments.append(answer)
        
        kfold_times.append(multiple_experts_time)
        answer = confidence_interval(multiple_experts_time)
        print(answer)
        confidence_interval_across_experiments_time.append(answer)

print(confidence_interval_across_experiments)
print(confidence_interval_across_experiments_time)
confidence_interval_across_experiments = np.asarray(confidence_interval_across_experiments)
confidence_interval_across_experiments_time = np.asarray(confidence_interval_across_experiments_time)
kfold_times = np.asarray(kfold_times)
np.save("mixtureOfExpertsResultsTimeAllTimes.npy", kfold_times)
np.save("mixtureOfExpertsResultsAccuracy.npy", confidence_interval_across_experiments)
np.save("mixtureOfExpertsResultsTime.npy", confidence_interval_across_experiments_time)