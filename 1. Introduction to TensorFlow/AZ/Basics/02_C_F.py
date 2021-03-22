# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # C to F by ML

# + id="w5bOPvFaQIqm"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# -

# ## Create DataSet

# + id="wEF4H2h3R_5o"
celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

# + colab={"base_uri": "https://localhost:8080/"} id="z1BLvZ_sSD-Q" outputId="442a2fe4-13cb-4618-86bf-fa4b77f4cbc7"
for i,c in enumerate(celsius_q):
  print(str(c) + " degrees Celsius = degrees Fahrenheit " + str(fahrenheit_a[i]))


# + colab={"base_uri": "https://localhost:8080/"} id="w0-C0HoRSG9-" outputId="8724160e-9216-42ab-c304-7693cf818b2a"

# -
# ## Model # 1

layer_0 = Dense(units=1, input_shape=[1])

model = Sequential()
model.add(layer_0)
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))
model.summary()

# + id="ay8_Sd4MShDO"
history = model.fit(x=celsius_q, y=fahrenheit_a, epochs=500, verbose=False)

# + colab={"base_uri": "https://localhost:8080/", "height": 296} id="I7pas988SOjO" outputId="2758d191-7b14-4afd-803a-90813ad5c553"
plt.xlabel('Epoch')
plt.ylabel("Loss")
plt.plot(history.history['loss'])

# -
# ### Prediction

# + colab={"base_uri": "https://localhost:8080/"} id="SyZ6kaQ6SQd2" outputId="e84ede3c-b3b1-46f5-b90d-ee4c3581d8a2"
print(model.predict([100.0]))

# + colab={"base_uri": "https://localhost:8080/"} id="elWiukimSSKe" outputId="ecb17893-07e6-4ca1-b0f8-b8a746e4d480"
print("These are the layer variables: " + str(layer_0.get_weights()))
# Note the formula is F = 1.8 * C + 32    , where weight is 1.8 and bias is 32

# +
print("End of the code")