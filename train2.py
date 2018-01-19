#!/usr/bin/env python

# This file is using found parameters to train a model with the whole dataset, and saves the model into a pickle file

import csv
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from math import floor
from random import shuffle
import time
import multiprocessing as mp
import pickle # To save and later load the trained model

x = []
y = []
indices_list = []

# Trains the final neural network
def train_final(x, y):
    clf = MLPClassifier(alpha = 0.01, learning_rate_init=0.01, max_iter=1000, hidden_layer_sizes=(100,100)) # Creates a classifier 
    clf.fit(x, y) # Train/fit using current set
    return clf;    


# Reads the CSV file and store its data in variables named x and y
def readCSV(filename):
    global x, y
    print('Processing CSV file')
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONE)
        data = np.array(list(reader)).astype("float")
        np.random.shuffle(data)
    
        x = data[:,[0,1,2,3,4]]
        y = data[:, -5:] #output 

def main():
    # Read the CSV File
    readCSV('data3_encoded_unique.csv')

    # Train model with found parameters and whole dataset
    clf = train_final(x, y)
    accuracy = clf.score(x, y)
    print("\nAccuracy of final neural network: .....", accuracy*100, "%\n")

    # Save model into a file
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    # To load use
    # with open('filename.pkl', 'rb') as f:
    # clf = pickle.load(f)


if __name__=='__main__':
    main()
