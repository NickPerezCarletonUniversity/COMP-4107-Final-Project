import matplotlib.pyplot as plt
import numpy as np


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



x = np.arange(2,11)
mixtureOfExperts = np.load("MixtureOfExperts/3EpochsTestResults/mixtureOfExpertsResultsAccuracy.npy")

plt.plot(x, mixtureOfExperts[:,0], color='red')
plt.plot(x, mixtureOfExperts[:,1], color='red')
plt.plot(x, mixtureOfExperts[:,2], color='red')

x = np.arange(2,11)
ensembleMethod = np.load("EnsembleMethod/3EpochsTestResults/ensembleMethodResultsAccuracy.npy")

plt.plot(x, ensembleMethod[:,0], color='blue')
plt.plot(x, ensembleMethod[:,1], color='blue')
plt.plot(x, ensembleMethod[:,2], color='blue')

x = np.arange(2,11)
ensembleMethod = np.load("EnsembleMethod/3EpochsTestResults/normalMethodResultsAccuracy.npy")

temp = ensembleMethod
for i in range(2,11-1):
    ensembleMethod = np.vstack((ensembleMethod, temp))

plt.plot(x, ensembleMethod[:,0], color='black')
plt.plot(x, ensembleMethod[:,1], color='black')
plt.plot(x, ensembleMethod[:,2], color='black')

plt.title('accuracies across different models trained')

plt.show()




x = np.arange(2,11)
ensembleMethod = np.load("EnsembleMethod/3EpochsTestResults/ensembleMethodResultsTime.npy")

plt.plot(x, ensembleMethod[:,0], color='blue')
plt.plot(x, ensembleMethod[:,1], color='blue')
plt.plot(x, ensembleMethod[:,2], color='blue')

x = np.arange(2,11)
mixtureOfExperts = np.load("MixtureOfExperts/3EpochsTestResults/mixtureOfExpertsResultsTime.npy")

plt.plot(x, mixtureOfExperts[:,0], color='red')
plt.plot(x, mixtureOfExperts[:,1], color='red')
plt.plot(x, mixtureOfExperts[:,2], color='red')

x = np.arange(2,11)
ensembleMethod = np.load("EnsembleMethod/3EpochsTestResults/normalMethodResultsTime.npy")

temp = ensembleMethod
for i in range(2,11-1):
    ensembleMethod = np.vstack((ensembleMethod, temp))

plt.plot(x, ensembleMethod[:,0], color='black')
plt.plot(x, ensembleMethod[:,1], color='black')
plt.plot(x, ensembleMethod[:,2], color='black')

plt.title('time to train across different models trained')

plt.show()






x = np.arange(2,11)
ensembleMethod = np.load("EnsembleMethod/3EpochsTestResults/ensembleMethodResultsTime.npy")

plt.plot(x, ensembleMethod[:,0], color='blue')
plt.plot(x, ensembleMethod[:,1], color='blue')
plt.plot(x, ensembleMethod[:,2], color='blue')

x = np.arange(2,11)
mixtureOfExperts = np.load("MixtureOfExperts/3EpochsTestResults/mixtureOfExpertsResultsTime.npy")

plt.plot(x, mixtureOfExperts[:,0], color='red')
plt.plot(x, mixtureOfExperts[:,1], color='red')
plt.plot(x, mixtureOfExperts[:,2], color='red')

x = np.arange(2,11)
ensembleMethod = np.load("EnsembleMethod/3EpochsTestResults/normalMethodResultsTime.npy")

temp = ensembleMethod
for i in range(2,11-1):
    ensembleMethod = np.vstack((ensembleMethod, temp))

plt.plot(x, ensembleMethod[:,0], color='black')
plt.plot(x, ensembleMethod[:,1], color='black')
plt.plot(x, ensembleMethod[:,2], color='black')

#this is the time showing the confidence interval for kmeans clustering
x = np.arange(2,11)
kmeans_times = np.load("MixtureOfExperts/kmeans_computation_time_per_fold.npy")
interval_for_kmeans = []
for i in kmeans_times:
    interval_for_kmeans.append(confidence_interval(i))

interval_for_kmeans = np.asarray(interval_for_kmeans)

plt.plot(x, interval_for_kmeans[:,0], color='green')
plt.plot(x, interval_for_kmeans[:,1], color='green')
plt.plot(x, interval_for_kmeans[:,2], color='green')

plt.title('time to train across different models trained')

plt.show()



#red is mixture of experts
#blue is ensemble method
#black is the normal, one model trained method
#the last graph shows time trained but with kmeans training time also shown in green



