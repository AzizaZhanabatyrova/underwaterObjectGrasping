#!/usr/bin/env python

# This file is using K-fold cross validation to obtain parameters of the neural network with a good accuracy

import csv
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from math import floor
from random import shuffle
import time
import string
import multiprocessing as mp
import pickle # To save and later load the trained model

x = []
y = []
indices_list = []


# Trains the neural network for a specific training and test set
def train(x, y, indices_list, i, q, avg_q):

    # Obtains input and outputs of training set and test set
    x_current_train, x_current_test = x[indices_list[i][0]], x[indices_list[i][1]]
    y_current_train, y_current_test = y[indices_list[i][0]], y[indices_list[i][1]]

    clf = MLPClassifier(alpha = 0.01, learning_rate_init=0.01, max_iter=1000, hidden_layer_sizes=(100,100))  # Creates a classifier aaa
    clf.fit(x_current_train, y_current_train) # Train/fit using current set
    
    net_output_test = clf.predict_proba(x_current_test)
    
    for output in net_output_test:
        max_prob_class = np.argmax(output)
        output = np.array([0,0,0,0,0,0,0])
        output[max_prob_class] = 1
        #print(output)
        
 
    current_accuracy = clf.score(x_current_test,y_current_test) # Obtain accuracy from the trained neural network
    
    # Write the accuracy to the 'q' queue
    q.put(current_accuracy)
    # Write the weighted accuracy to the 'avg_q' queue
    avg_q.put(current_accuracy*len(indices_list[i][0])) 


# Reads the CSV file and store its data in variables named x and y
def readCSV(filename):
    global x, y
    print('Processing CSV file')
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONE)
        data = np.array(list(reader)).astype('float')
        np.random.shuffle(data)
        #x = np.concatenate((data[:,4:5], data[:, :3]))#input 
        x = data[:,[0,1,2,3,4]]
        y = data[:, -5:] #output 
        

# Use KFold splitting method
def useKFold(k):
    global indices_list
    print('Starting KFold')
    kf = KFold(n_splits = k, random_state=None, shuffle=False) # Create sets
    indices_list = [(train_indices, test_indices) for train_indices, test_indices in kf.split(x)]


def main():
    # Read the CSV File
    readCSV('data3_encoded_unique.csv')

    # Use KFold splitting
    k = 10
    useKFold(k)

    # Start measuring time
    start_time = time.time()

    # Create FIFO queues for interprocess communication
    q = mp.Queue()
    avg_q = mp.Queue()

    # Processes list, accuracies list and weighted accuracies list
    procs, accuracies, weighted_accuracies = [], [], []

    # Define maximum number of simultaneous processes
    nprocs = 2
    
    # Parameters for process creation loop
    m = 0
    n = nprocs

    # Process creation loop
    for j in range(6):
        # Creates 'nprocs' simultaneous processes and start them
        for i in range(m, n):
            p = mp.Process(target=train, args=(x, y, indices_list, i, q, avg_q))
            procs.append(p)
            p.start()
            print('Training ', i, ' started.')

        # Communicate with the previously created processes to obtain the accuracy calculated for the training
        for i in range(m, n):
            r = q.get()
            accuracies.append(r)
            s = avg_q.get()
            weighted_accuracies.append(s)
            print('Partial accuracy obtained from training: ', r*100, '%')
      
        # Join all processes
        for p in procs:
            p.join()

        # Empty the process list
        procs = []
        
        # Define new values for the parameters for process creation loop
        m = n 
        n = m + nprocs
        if (n > k):
            n = k

    # Finishes measuring time
    end_time = time.time()

    # Calculates weighted average accuracy
    avg_accuracy = 0
    avg_accuracy = sum(weighted_accuracies)/(sum([len(indices_list[i][0]) for i in range(0,k)]))
    print('Total time (in seconds): ', end_time - start_time)
    print('Training finished. Average accuracy: ', avg_accuracy*100, '%')


if __name__=='__main__':
    main()
