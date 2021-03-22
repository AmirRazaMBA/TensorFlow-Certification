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

# # CNN - Example 01

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

# ### Load Keras Dataset

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# #### Visualize data

print(x_train.shape)
single_image = x_train[0]
print(single_image.shape)
plt.imshow(single_image)

# ### Pre-Process data

# #### One Hot encode

# Make it one hot encoded otherwise it will think as a regression problem on a continuous axis
from tensorflow.keras.utils import to_categorical
print("Shape before one hot encoding" +str(y_train.shape))
y_example = to_categorical(y_train)
print(y_example)
print("Shape after one hot encoding" +str(y_train.shape))
y_example[0]

y_cat_test = to_categorical(y_test,10)
y_cat_train = to_categorical(y_train,10)

# #### Normalize the images

x_train = x_train/255
x_test = x_test/255

scaled_single = x_train[0]
plt.imshow(scaled_single)

# #### Reshape the images

# Reshape to include channel dimension (in this case, 1 channel)
# x_train.shape
x_train = x_train.reshape(60000, 28, 28, 1)  
x_test = x_test.reshape(10000,28,28,1)

# ### Model # 1

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4), input_shape=(28, 28, 1), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))
# FLATTEN IMAGES FROM 28 by 28 to 764 BEFORE FINAL LAYER
model.add(Flatten())
# 128 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
model.add(Dense(128, activation='relu'))
# LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES
model.add(Dense(10, activation='softmax'))

# Notes : If y is not one hot coded then loss= sparse_categorical_crossentropy

model.compile(loss='categorical_crossentropy',  
              optimizer='adam',
              metrics=['accuracy', 'categorical_accuracy']) 
              # we can add in additional metrics https://keras.io/metrics/

model.summary()

# #### Add Early Stopping

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=2)

# ##### Training using one hot encoding

history = model.fit(x_train,y_cat_train,
                      epochs=10,
                      validation_data=(x_test,y_cat_test),
                      callbacks=[early_stop])

# #### Save model

# Saving model
from tensorflow.keras.models import load_model
model_file = 'D:\\Sandbox\\Github\\MODELS\\' + '01_mnist.h5'
model.save(model_file)

# #### Retreive model

# Retrieve model
# model = load_model(model_file)

# #### Evaluate

# Rule of thumb
# 1. High Bias                        accuracy = 80% val-accuracy = 78%   (2%  gap)
# 2. High Variance                    accuracy = 98% val-accuracy = 80%   (18% gap)
# 3. High Bias and High Variance      accuracy = 80% val-accuracy = 60%   (20% gap)
# 4. Low Bias and Low Variance        accuracy = 98% val-accuracy = 96%   (2%  gap)

# #### Eval - Train

model.metrics_names

pd.DataFrame(history.history).head()
#pd.DataFrame(model.history.history).head()

# pd.DataFrame(history.history).plot()

losses = pd.DataFrame(history.history)

losses[['loss','val_loss']].plot()

losses[['accuracy','val_accuracy']].plot()

# Plot loss per iteration
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()

# Plot accuracy per iteration
plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()

# #### Eval - Test

test_metrics = model.evaluate(x_test,y_cat_test,verbose=1)

print('Loss on test dataset:', test_metrics[0])
print('Accuracy on test dataset:', test_metrics[1])

print("Loss and Accuracy on Train dataset:")

pd.DataFrame(history.history).tail(1)

# As it turns out, the accuracy on the test dataset is smaller than the accuracy on the training dataset. 
# This is completely normal, since the model was trained on the `train_dataset`. 
# When the model sees images it has never seen during training, (that is, from the `test_dataset`), 
# we can expect performance to go down. 

# #### Prediction

y_prediction = np.argmax(model.predict(x_test), axis=-1)

# #### Reports

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test, y_prediction))
print(confusion_matrix(y_test, y_prediction))

# Recall (sensivity)   : Fraud detection recall because you want to catch FN (real fraud guys)
# Precision (specificity): Sentiment analysis precision is important. You want to catch all feeling FP ()
# F1 score  : Higher is better to compare two or more models
# accuracy  : higher is better
# error     : 1 - accuracy
# Ideally, We want both Precision & Recall to be 1 but it is a zero-sum game. You can't have both 

import seaborn as sns
plt.figure(figsize=(10,6))
sns.heatmap(confusion_matrix(y_test,y_prediction),annot=True)

# #### Predictions go wrong!

# Show some misclassified examples
misclassified_idx = np.where(y_prediction != y_test)[0]
i = np.random.choice(misclassified_idx)
plt.imshow(x_test[i].reshape(28,28), cmap='gray')
plt.title("True label: %s Predicted: %s" % (y_test[i], y_prediction[i]));
plt.show()

# ### Model # 2

model_2 = Sequential()

model_2.add(Conv2D(filters=32, kernel_size=(4,4), input_shape=(28, 28, 1), activation='relu',))
model_2.add(MaxPool2D(pool_size=(2, 2)))
model_2.add(Flatten())
model_2.add(Dense(128, activation='relu'))
model_2.add(Dense(10, activation='sigmoid'))
model_2.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']) 
model_2.summary()

# #### Add Early Stopping

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=2)

history_2 = model_2.fit(x_train,y_train,
                      epochs=10,
                      validation_data=(x_test,y_test),
                      callbacks=[early_stop])


lossess = history_2.history
lossess.keys()

pd.DataFrame(lossess).head()

pd.DataFrame(lossess).tail(1)

losses[['loss','val_loss']].plot()

losses[['accuracy','val_accuracy']].plot()

# #### Prediction

y_prediction = np.argmax(model_2.predict(x_test), axis=-1)

# #### Reports

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test, y_prediction))
print(confusion_matrix(y_test, y_prediction))

# Recall (sensivity)   : Fraud detection recall because you want to catch FN (real fraud guys)
# Precision (specificity): Sentiment analysis precision is important. You want to catch all feeling FP ()
# F1 score  : Higher is better to compare two or more models
# accuracy  : higher is better
# error     : 1 - accuracy
# Ideally, We want both Precision & Recall to be 1 but it is a zero-sum game. You can't have both 

import seaborn as sns
plt.figure(figsize=(10,6))
sns.heatmap(confusion_matrix(y_test,y_prediction),annot=True)

# #### Predictions go wrong!

# Show some misclassified examples
misclassified_idx = np.where(y_prediction != y_test)[0]
i = np.random.choice(misclassified_idx)
plt.imshow(x_test[i].reshape(28,28), cmap='gray')
plt.title("True label: %s Predicted: %s" % (y_test[i], y_prediction[i]));
plt.show()
# #### Final thoughts

# Which model is better? Both models are same only loss types and y train are different

# Rule of thumb
# 1. High Bias                        accuracy = 80% val-accuracy = 78%   (2%  gap)
# 2. High Variance                    accuracy = 98% val-accuracy = 80%   (18% gap)
# 3. High Bias and High Variance      accuracy = 80% val-accuracy = 60%   (20% gap)
# 4. Low Bias and Low Variance        accuracy = 98% val-accuracy = 96%   (2%  gap)

print("Percentage of wrong predcitions : " + str(len(misclassified_idx)/len(y_prediction)*100) + " %")
print("Models maximum accuracy            : " + str(np.max(history.history['accuracy'])*100) + " %")
print("Models maximum validation accuracy : " + str(np.max(history.history['val_accuracy'])*100) + " %")

# Model has Low Bias and Low Variance with less than 1% gap. It is already at 99% accuracy

