#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

for i, c in enumerate(celsius_q):
    print(str(c) + " degrees Celsius = degrees Fahrenheit " + str(fahrenheit_a[i]))

layer_0 = Dense(units=1, input_shape=[1])

model = Sequential()
model.add(layer_0)
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))
model.summary()

history = model.fit(x=celsius_q, y=fahrenheit_a, epochs=500, verbose=False)

plt.xlabel('Epoch')
plt.ylabel("Loss")
plt.plot(history.history['loss'])

print(model.predict([100.0]))

print("These are the layer variables: " + str(layer_0.get_weights()))
# Note the formula is F = 1.8 * C + 32    , where weight is 1.8 and bias is 32
