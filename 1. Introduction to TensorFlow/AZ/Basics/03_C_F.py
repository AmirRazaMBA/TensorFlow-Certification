# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:nomarker
#     text_representation:
#       extension: .py
#       format_name: nomarker
#       format_version: '1.0'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



# # C to F from ML Here

celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

#for i,c in enumerate(celsius_q):
  #print(str(c) + " degrees Celsius = degrees Fahrenheit " + str(fahrenheit_a[i]))

layer_0 = Dense(units=4, input_shape=[1])
layer_1 = Dense(units=4)
layer_2 = Dense(units=1)

model = Sequential()
model.add(layer_0)
model.add(layer_1)
model.add(layer_2)

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))
model.summary()

#  Layer 0 : weights 1 x 4 and bias 4  ; Layer 1 : weights 4 x 4 and bias 4 ; Layer 3 : weights 4 x 1 and bias = 1

history = model.fit(x=celsius_q, y=fahrenheit_a, epochs=500, verbose=False)

# Check any change in the loss when more layers are added compared to one layer in the last ex?
plt.xlabel('Epoch')
plt.ylabel("Loss")
plt.plot(history.history['loss'])


# # Predictions

print(model.predict([100.0]))

# Note the formula is F = 1.8 * C + 32    , where weight is 1.8 and bias is 32 if only one layer and one unit
print("These are the layer_0 variables: " + str(layer_0.get_weights()) + "\n")
print("These are the layer_1 variables: " + str(layer_1.get_weights()) + "\n")
print("These are the layer_2 variables: " + str(layer_2.get_weights()) + "\n")
