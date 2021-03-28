# -*- coding: utf-8 -*-
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

# # CNN - Example 03

import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense

# ## Load Tensorflow Dataset (tfds)

# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)

# Import TensorFlow Datasets
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# ### Get train/test dataset

dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

train_dataset, test_dataset = dataset['train'], dataset['test']

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def get_name(id):
    return class_names[id]


num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:     {}".format(num_test_examples))

tfds.as_dataframe(train_dataset.take(5), metadata)


# ### Normalize

# # Preprocess the data

# The value of each pixel in the image data is an integer in the range `[0,255]`.
# For the model to work properly, these values need to be normalized to the range `[0,1]`.
# So here we create a normalization function, and then apply it to each image in the test and train datasets.

def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


# # The map function applies the normalize function to each element in the train
# # and test datasets
# train_dataset = train_dataset.map(normalize)
# test_dataset = test_dataset.map(normalize)

# The first time you use the dataset, the images will be loaded from disk
# Caching will keep them in memory, making training faster. Moving cache below
# train_dataset = train_dataset.cache()
# test_dataset = test_dataset.cache()

# ### Explore the data

# Take a single image, and remove the color dimension by reshaping
for image, label in test_dataset.take(1):
    break
image = image.numpy().reshape((28, 28))
plt.imshow(image, cmap=plt.cm.binary)
print(get_name(label))

# Display the first 25 images from the *training set* and display the class name below each image. Verify that the
# data is in the correct format and we're ready to build and train the network.

# ### Display few images

plt.figure(figsize=(10, 10), facecolor="red")
i = 0
for (image, label) in test_dataset.take(25):
    image = image.numpy().reshape((28, 28))
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
    i += 1
plt.show()

# ## Model # 1

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu, input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2), strides=2))
model.add(Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu))
model.add(MaxPooling2D((2, 2), strides=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

# Loss function — An algorithm for measuring how far the model's outputs are from the desired output. The goal of
# training is this measures loss. Optimizer —An algorithm for adjusting the inner parameters of the model in order to
# minimize loss. Metrics —Used to monitor the training and testing steps. The following example uses accuracy,
# the fraction of the images that are correctly classified.

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# ### Train the model
#
# First, we define the iteration behavior for the train dataset: 1. Repeat forever by specifying `dataset.repeat()`
# 2. The `dataset.shuffle(60000)` randomizes the order so our model cannot learn anything from the order of the
# examples. 3. And `dataset.batch(32)` tells `model.fit` to use batches of 32 images and labels when updating the
# model variables. 4. The `epochs=5` parameter limits training to 5 full iterations of the training dataset,
# so a total of 5 * 60000 = 300000 examples.
#
# (Don't worry about `steps_per_epoch`, the requirement to have this flag will soon be removed.)

# #### Normalize, Cache, Repeat, Shuffle Batch and Prefetch.

BATCH_SIZE = 32
#AUTOTUNE = tensorflow.core.experimental.AUTOTUNE
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.map(normalize).cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE).prefetch(AUTOTUNE)
test_dataset = test_dataset.map(normalize).batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)

history = model.fit(train_dataset, epochs = 100, steps_per_epoch=math.ceil(num_train_examples / BATCH_SIZE))

# ### Save the trained model

model_filename = "colab_model/02_fashion_mnist_tfds_model"

model.save(model_filename)


print(tf.__version__)

# !ls

# !cp -r colab_model/02_fashion_mnist_tfds_model /content/drive/MyDrive/ColabModels/

# ### Evaluate

# #### Eval - Train

model.metrics_names

pd.DataFrame(history.history).head(10)

pd.DataFrame(history.history).plot()

losses = pd.DataFrame(history.history)

print(losses)

losses[['loss', 'accuracy']].plot()

# #### Eval - Test

new_model = tf.keras.models.load_model(saved_model_filename)
new_model.summary()
test_loss, test_accuracy = new_model.evaluate(test_dataset, steps=math.ceil(num_test_examples / 32))

print('Loss on test dataset:', test_loss)
print('Accuracy on test dataset:', test_accuracy)

# As it turns out, the accuracy on the test dataset is smaller than the accuracy on the training dataset. 
# This is completely normal, since the model was trained on the `train_dataset`. 
# When the model sees images it has never seen during training, (that is, from the `test_dataset`), 
# we can expect performance to go down. 

print("Loss and Accuracy on Train dataset:")

pd.DataFrame(history.history).tail()

# ### Predictions

print(len(test_dataset) / BATCH_SIZE)

for test_images, test_labels in test_dataset.take(1):
    x_test = test_images.numpy()
    y_test = test_labels.numpy()

y_prediction = np.argmax(new_model.predict(x_test), axis=-1)

print(len(y_test))

print(len(y_prediction))

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_prediction))
print(confusion_matrix(y_test, y_prediction))

plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix(y_test, y_prediction), annot=True)

# ### Predictions go wrong!

# Show some misclassified examples
misclassified_idx = np.where(y_prediction != y_test)[0]
i = np.random.choice(misclassified_idx)
plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
# plt.title("True label: %s Predicted: %s" % (y_test[i], y_prediction[i]));
t = get_name(y_test[i])
p = get_name(y_prediction[i])
plt.title("True label: %s Predicted: %s" % (t, p));

print("Percentage of wrong predications : " + str(len(misclassified_idx) / len(y_prediction) * 100) + " %")
print("Models maximum accuracy            : " + str(np.max(history.history['accuracy']) * 100) + " %")
# print("Models maximum validation accuracy : " + str(np.max(history.history['val_accuracy'])*100) + " %")

# ### Additonal Files(s) to test the model

# Grab an image from the test dataset
img = test_images[0]

print(img.shape)

img = np.array([img])

print(img.shape)

predictions_single = new_model.predict(img)

print(predictions_single)

np.argmax(predictions_single[0])

# ### Exercises
#
# Experiment with different models and see how the accuracy results differ. In particular change the following
# parameters: *   Set training epochs set to 1 *   Number of neurons in the Dense layer following the Flatten one.
# For example, go really low (e.g. 10) in ranges up to 512 and see how accuracy changes *   Add additional Dense
# layers between the Flatten and the final Dense(10), experiment with different units in these layers *   Don't
# normalize the pixel values, and see the effect that has
#
#
# Remember to enable GPU to make everything run faster (Runtime -> Change runtime type -> Hardware accelerator -> GPU).
# Also, if you run into trouble, simply reset the entire environment and start from the beginning:
# *   Edit -> Clear all outputs
# *   Runtime -> Reset all runtimes
