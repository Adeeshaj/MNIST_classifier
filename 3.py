from keras import models, layers
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

noise_factor = 0.25
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy =  x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

autoencoder = models.Sequential()
#encoding
autoencoder.add(layers.Conv2D(80, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
autoencoder.add(layers.Conv2D(40, kernel_size=(3, 3), activation='relu'))

#decoding

autoencoder.add(layers.Conv2DTranspose(40, kernel_size=(3, 3), activation='relu'))
autoencoder.add(layers.Conv2DTranspose(80, kernel_size=(3, 3), activation='relu'))

autoencoder.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

x_train_noisy = x_train_noisy.reshape((60000, 28, 28, 1))
x_train_noisy = x_train_noisy.astype('float32') / 255 # rescale pixel values from range [0, 255] to [0, 1]
x_train = x_train.reshape((60000, 28, 28, 1))
x_train = x_train.astype('float32') / 255

autoencoder_history = autoencoder.fit(x_train_noisy, x_train, epochs=2)

autoencoded_train = autoencoder.predict(x_train_noisy)
autoencoded_train = autoencoded_train.reshape((60000, 28, 28, 1))
autoencoded_train = autoencoded_train.astype('float32') / 255

model = models.Sequential()
model.add(layers.Conv2D(80, (12, 12), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((3, 3)))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['accuracy'])


history = model.fit(autoencoded_train, y_train, epochs=10)
