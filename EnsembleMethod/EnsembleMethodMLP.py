#portions of this code were taken from the in class MLP.py example
import tensorflow as tf
import numpy as np
import time
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

average_model_accuracy = 0

epochs_per_model = 3

batch_size = 128

number_of_folds = 10

average_model_accuracies = []
ensemble_method_accuracies = []
ensemble_method_time = []

models_to_train_begin = 2
models_to_train_end = 10

#models_to_train_begin = 1
#models_to_train_end = 1

confidence_interval_across_experiments = []
confidence_interval_across_experiments_time = []
aggregated_time_across_models = 0


# Launch the graph in a session
with tf.Session() as sess:
    for n in range(models_to_train_begin, models_to_train_end + 1):
        models_to_train = n
        average_model_accuracies = []
        ensemble_method_accuracies = []
        ensemble_method_time = []
        for fold in range(0, number_of_folds):
            print("fold number: " + str(fold))
            size_of_fold = int(len(total_data_set)/number_of_folds)
            trX_for_models = np.vstack((total_data_set[0:fold*size_of_fold], total_data_set[(fold + 1)*size_of_fold:len(total_data_set)]))
            teX = total_data_set[fold*size_of_fold:(fold + 1)*size_of_fold]

            trY_for_models = np.vstack((total_label_set[0:fold*size_of_fold], total_label_set[(fold + 1)*size_of_fold:len(total_label_set)]))
            teY = total_label_set[fold*size_of_fold:(fold + 1)*size_of_fold]

            print("x_train shape:", trX_for_models.shape, "y_train shape:", trY_for_models.shape)
            print("x_test shape:", teX.shape, "y_test shape:", teY.shape)

            average_model_accuracy = 0
            aggregated_time_across_models = 0

            for z in range(0,models_to_train):

                training_per_model = int(len(trX_for_models)/models_to_train)
                trX = trX_for_models[z*training_per_model:(z+1)*training_per_model]
                trY = trY_for_models[z*training_per_model:(z+1)*training_per_model]
                print(trX.shape)
                print(trY.shape)

                size_h1 = tf.constant(625, dtype=tf.int32)
                size_h2 = tf.constant(300, dtype=tf.int32)

                X = tf.placeholder("float", [None, 784])
                Y = tf.placeholder("float", [None, 10])

                w_h1 = init_weights([784, size_h1]) # create symbolic variables
                w_h2 = init_weights([size_h1, size_h2])
                w_o = init_weights([size_h2, 10])

                py_x = model(X, w_h1, w_h2, w_o)

                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute costs
                train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
                predict_op = tf.argmax(py_x, 1)
                tf.global_variables_initializer().run()
                for i in range(epochs_per_model):
                    time_start = time.process_time()
                    for start, end in zip(myRange(0, len(trX), batch_size), myRange(batch_size, len(trX)+1, batch_size)):
                        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
                    time_stop = time.process_time()
                    aggregated_time_across_models = aggregated_time_across_models + (time_stop - time_start)
                    print(i, np.mean(np.argmax(teY, axis=1) ==
                                     sess.run(predict_op, feed_dict={X: teX})))
                average_model_accuracy = np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={X: teX}))/models_to_train + average_model_accuracy
                if z == 0:
                    prediction = sess.run(py_x, feed_dict={X: teX})
                else:
                    prediction = np.add(prediction, sess.run(py_x, feed_dict={X: teX}))

            print("average accuracy of all the models: " + str(average_model_accuracy))
            print("accuracy of the predictions of all the models combined: " + str(np.mean(np.argmax(teY, axis=1) == np.argmax(prediction, axis=1)))) 
            average_model_accuracies.append(average_model_accuracy)
            ensemble_method_accuracies.append(np.mean(np.argmax(teY, axis=1) == np.argmax(prediction, axis=1)))
            ensemble_method_time.append(aggregated_time_across_models)

        answer = confidence_interval(ensemble_method_accuracies)
        print(answer)
        confidence_interval_across_experiments.append(answer)
        
        answer = confidence_interval(ensemble_method_time)
        print(answer)
        confidence_interval_across_experiments_time.append(answer)

print(confidence_interval_across_experiments)
confidence_interval_across_experiments = np.asarray(confidence_interval_across_experiments)
np.save("ensembleMethodResultsAccuracy.npy", confidence_interval_across_experiments)
#np.save("normalMethodResultsAccuracy.npy", confidence_interval_across_experiments)

print(confidence_interval_across_experiments_time)
confidence_interval_across_experiments_time = np.asarray(confidence_interval_across_experiments_time)
np.save("ensembleMethodResultsTime.npy", confidence_interval_across_experiments_time)
#np.save("normalMethodResultsTime.npy", confidence_interval_across_experiments_time)
