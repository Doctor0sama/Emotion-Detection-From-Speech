import pandas as pd
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
import math

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

round_half_up_v = np.vectorize(round_half_up)

features = pd.read_csv("features.csv")
labels = pd.read_csv("labels.csv")
print(features)
print(labels)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
print(encoded_Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
print(dummy_y)
print(dummy_y.shape)
print(labels)

X = np.asarray(features)
dummy_y = np.asarray(dummy_y)
X_train, X_notTrain, y_train, y_notTrain = train_test_split(X, dummy_y, test_size=0.4, random_state=9)
X_test, X_val, y_test, y_val = train_test_split(X_notTrain, y_notTrain, test_size=0.5, random_state=42)

X_train = np.reshape(X_train, (X_train.shape[0], 2, 10))
X_test = np.reshape(X_test, (X_test.shape[0], 2, 10))
X_val = np.reshape(X_val, (X_val.shape[0], 2, 10))

model = load_model("Model2.h5")
model.summary()

predicted_y = model.predict(X_test, batch_size=16)

print("Predicted y before rounding: ")
print(predicted_y)

print("Predicted y after rounding: ")
print(round_half_up_v(predicted_y))

predicted_y01 = round_half_up_v(predicted_y)

print("Actual y: ")
print(y_test)

print("\nEvaluating : ")
loss, accuracy = model.evaluate(X_test, y_test, batch_size=16)
print()
print("Final Accuracy : ", accuracy)
print("Final Loss : ", loss)
