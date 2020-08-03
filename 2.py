from keras import models, layers
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

#load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#the CNN model
model = models.Sequential()
model.add(layers.Conv2D(80, (12, 12), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((3, 3)))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#add noise
noise_factor = 0.25
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy =  x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)


x_train_noisy = x_train_noisy.reshape((60000, 28, 28, 1))
x_train_noisy = x_train_noisy.astype('float32') / 255 # rescale pixel values from range [0, 255] to [0, 1]

x_test_noisy = x_test_noisy.reshape((10000, 28, 28, 1))
x_test_noisy = x_test_noisy.astype('float32') / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


history = model.fit(x_train_noisy, y_train, epochs=10, batch_size=64, validation_data=(x_test_noisy, y_test))
