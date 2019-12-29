import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.layers import Dense, concatenate
from keras.layers import LSTM
from keras.utils import to_categorical
from keras.layers import Input, Dropout, Activation
from keras.layers import Conv1D, GlobalAveragePooling1D, Permute, BatchNormalization
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils, to_categorical


tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
K.clear_session()

X = pd.read_csv("features.csv")
y = pd.read_csv("labels.csv")

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
print(encoded_Y)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
print(dummy_y)
print(dummy_y.shape)
print(y)

X = np.asarray(X)
dummy_y = np.asarray(dummy_y)
X_train, X_notTrain, y_train, y_notTrain = train_test_split(X, dummy_y, test_size=0.4, random_state=9)
X_test, X_val, y_test, y_val = train_test_split(X_notTrain, y_notTrain, test_size=0.5, random_state=42)

X_train = np.reshape(X_train, (X_train.shape[0], 2, 10))
X_test = np.reshape(X_test, (X_test.shape[0], 2, 10))
X_val = np.reshape(X_val, (X_val.shape[0], 2, 10))

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#Creating model
ip = Input(shape=(2, 10))

x = LSTM(8)(ip)
x = Dropout(0.5)(x)

y = Permute((2, 1))(ip)
y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
y = BatchNormalization()(y)
y = Activation('relu')(y)

y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
y = BatchNormalization()(y)
y = Activation('relu')(y)

y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
y = BatchNormalization()(y)
y = Activation('relu')(y)

y = GlobalAveragePooling1D()(y)
y = Dense(2, activation='sigmoid')(y)
x = concatenate([x, y])

out = Dense(2, activation='softmax')(x)

model = Model(ip, out)

model.summary()

learning_rate = 1e-3

reduce_lr = ReduceLROnPlateau(monitor='loss', patience=10, mode='auto',
                                   cooldown=0, min_lr=1e-4, verbose=2)

optm = Adam(lr=learning_rate)

model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=10, epochs=100, callbacks=[reduce_lr],
             verbose=2, validation_data=(X_val, y_val))

print("\nEvaluating : ")
loss, accuracy = model.evaluate(X_test, y_test, batch_size=16)
print()
print("Final Accuracy : ", accuracy)
print("Final Loss : ", loss)

predicted_y = model.predict(X_test, batch_size=16)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save("Model.h5")