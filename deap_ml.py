import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sc
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils, plot_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# Function to find the power in a series of signals
def find_power(data):
    sum_of_squares = 0
    #Find the sum of squares of all the values
    for value in data:
        sum_of_squares += (value * value)

    return sum_of_squares/len(data)

def find_valence_label(int_valence):
    
    if int_valence <= 9.0 and int_valence >= 6.0:
        return "HV"
    elif int_valence < 6.0 and int_valence > 4.0:
        return "MV"
    elif int_valence <= 4.0 and int_valence >= 0.0:
        return "LV"
    else:
        print("Error was made in trying to label a valence:")
        print("\tValence Value: " + str(int_valence))
        return "EV"

def make_valence_labels(valence_array):
    valence_label_array = []

    # Take every valence rating of the subject and assign it a label (High Valence, Middle Valence, Low Valence)
    for valence_rating in valence_array:
        valence_label_array.append(find_valence_label(valence_rating))

    return valence_label_array

# Open file of the first patient, put it in numpy table
s1_file = open("./data_preprocessed_python/s01.dat", "rb")
s1 = pickle.load(s1_file,fix_imports=True, encoding="latin1")

s1_data = np.array(s1["data"])
s1_label = np.array(s1["labels"])

s1_int_valence = s1_label[:, 0]
# Create array with the categories of Valence
s1_labeled_valence = make_valence_labels(s1_int_valence)

# Create 2D array where the index is (video, channel) and the value is the entropy of the video and channel
s1_power = np.zeros(shape=(40,40))

# Find power for each video (x) and each channel (y)
for x in range(0,40):
    for y in range(0,40):
        s1_power[x, y] = find_power(s1_data[x,y,1664:])

# Make the Classifier
start_time = time.time()

# Fix random seed for reproductibility
seed = 100
np.random.seed(seed)

# Take X, the parameters to the model, and Y, the categories we are trying to fit the model to predict
X = s1_power
Y = s1_labeled_valence

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# Conert integers to dummy variables
category_y = np_utils.to_categorical(encoded_Y)

# create model
model = Sequential()
model.add(Conv1D(32, (3), input_shape=(40,8064)))
model.add(Dense(3, activation='softmax'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(s1_data, category_y, validation_split=0.1, epochs=100, batch_size=36, shuffle=True)

#print(model.get_layer(index=0).get_weights())

total_time = time.time() - start_time
print("Total Time: ", total_time)