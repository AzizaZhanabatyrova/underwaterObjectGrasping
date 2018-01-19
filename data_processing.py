#!/usr/bin/env python

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

def readCSV(filename):
    global data, unique_data
    print('Processing CSV file')
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONE)
        data = np.array(list(reader)).astype('float')
        # Consider only first three descriptors
        #cut_data = data[:,[0,1,2,3,4,5,6,7,8,9,10,11]]
        unique_data = np.unique(data, axis = 0)
    print('Done')


def saveCSV(filename):
    global unique_data
    print('Saving CSV file')
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_NONE, lineterminator='\n')
        
        for row in unique_data:
            writer.writerow(list(row))
        
def main():
    readCSV('data3_encoded.csv')
    saveCSV('data3_encoded_unique.csv')
    
    
if __name__=='__main__':
    main()
