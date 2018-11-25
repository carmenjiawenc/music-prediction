import keras
import pandas as pd
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np

le = preprocessing.LabelEncoder()

df_x = pd.read_csv('features_cat_6.csv')
X = df_x.iloc[:,1:519].values
y = df_x.iloc[:,520].values

pca = PCA(n_components = 'mle', svd_solver = 'full')
X = pca.fit_transform(X)

le.fit(y)
y = le.transform(y)
y = np_utils.to_categorical(y)
print ("Label Encoded")

(trainData, testData, trainLabels, testLabels) = train_test_split(X, y, test_size=0.25, random_state=42)

model = Sequential()
model.add(Dense(30, input_dim=X.shape[1], init="uniform",activation="relu"))
model.add(Dense(40, activation="relu", kernel_initializer="uniform"))
model.add(Dense(20, activation="relu", kernel_initializer="uniform"))
model.add(Dense(6))

model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

learning_rate_function = ReduceLROnPlateau(monitor='val_acc',patience = 3, verbose = 1, factor = 0.7, min_lr = 0.00001)

history = model.fit(trainData, trainLabels, epochs=65, validation_data=(testData, testLabels)
                    ,batch_size=128, verbose=2, shuffle = False, callbacks=[learning_rate_function])

model.save_weights('music_weight_cat6.h5')