# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% colab={"base_uri": "https://localhost:8080/"} id="PUNO2E6SeURH" outputId="f6cf9813-f481-413c-f991-75022976089f"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

Xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)

tf.keras.backend.clear_session()

model = tf.keras.Sequential()
model.add(keras.layers.Dense(units=1, input_shape=[1]))

# model.compile(optimizer='sgd', loss='mse')
# model.compile(optimizer='RMSprop', loess='mse' )
model.compile(optimizer='adam', loss='mse')

model.summary()

print('ok')

# %% [markdown]
# # Header 1

# %% [markdown]
# ## Header 2

# %% colab={"base_uri": "https://localhost:8080/"} id="5r5mt0chNAv1" outputId="45a03db5-3fef-4e0a-8181-379121662fdf"
history = model.fit(Xs, ys, epochs=1000, verbose=0)

# %% colab={"base_uri": "https://localhost:8080/", "height": 282} id="pbecd6AxKQae" outputId="0dcde4a5-2043-4eb7-869a-632e99fecf79"
# Plot the loss
plt.plot(history.history['loss'], label='loss')

# %% colab={"base_uri": "https://localhost:8080/"} id="K-zVJG0_Lztr" outputId="d59b56a1-1262-4936-ee91-66b781b2a22e"
# Get the slope and bias of the line
w, b = model.layers[0].get_weights()
print('w : ' + str(w))
print('b : ' + str(b))

# %% colab={"base_uri": "https://localhost:8080/"} id="SREFOyWALjWD" outputId="8f8683f9-c106-4b64-e377-9ffce639d514"
print(model.predict([5.0]))



import math
print("This is the end of the file")

# %% [markdown]
# #### Conclusion
