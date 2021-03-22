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

# ## CNN Example

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
print('Hello world')
# ## Load Dataset

# ## This code is for Google Colab to retreive the files.
# ///
# os.environ['KAGGLE_CONFIG_DIR'] = '/content'  # remove the full path
# # !kaggle datasets download -d ansubkhan/malaria-detection  # https://www.kaggle.com/ansubkhan/malaria-detection
# https://www.kaggle.com/syedamirraza/malaria-cell-image
#
# # unzip and remove the zip
# # !unzip \*.zip && rm *.zip
#
# my_data_dir = '/content/Malaria Detection/cell image'
# print(os.listdir(my_data_dir) ) # returns 'test', and 'train
# ///

my_data_dir = 'D:\\Sandbox\\GitHub\\DATA\\cell_images' 
print(os.listdir(my_data_dir) ) # returns 'test', and 'train

test_path = my_data_dir+'\\test\\'
train_path = my_data_dir+'\\train\\'

print(os.listdir(test_path))
print(os.listdir(train_path))
print(os.listdir(train_path+'\\parasitized')[0])

infected_cell_path = train_path+'\\parasitized'+'\\C100P61ThinF_IMG_20150918_144104_cell_162.png'
infected_cell= imread(infected_cell_path)
print(infected_cell.shape)
plt.imshow(infected_cell)

uninfected_cell_path = train_path+'\\uninfected\\'+os.listdir(train_path+'\\uninfected')[0]
uninfected_cell = imread(uninfected_cell_path)
print(uninfected_cell.shape)
plt.imshow(uninfected_cell)

# **Let's check how many images there are.**

# ## View data

# Let's check how many images there are.

print(len(os.listdir(train_path+'\\parasitized')))
print(len(os.listdir(train_path+'\\uninfected')))

# Let's find out the average dimensions of these images
print(uninfected_cell.shape)
print(infected_cell.shape)

# Issue size is not the same.
# One option: https://stackoverflow.com/questions/1507084/how-to-check-dimensions-of-all-images-in-a-directory-using-python
dim1 = []
dim2 = []
for image_filename in os.listdir(test_path+'\\uninfected'):
    
    img = imread(test_path+'\\uninfected'+'\\'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)

# ## View data

sns.jointplot(dim1,dim2)

# ### Make the image size almost similar

print(np.mean(dim1))
print(np.mean(dim2))

new_image_shape = (130,130,3)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )

plt.imshow(infected_cell)

# show so random transformed images
plt.imshow(image_gen.random_transform(infected_cell))

# show so random transformed images
plt.imshow(image_gen.random_transform(infected_cell))

plt.imshow(image_gen.random_transform(infected_cell))

# ## MAIN CODE for TF learning

image_gen.flow_from_directory(train_path)

image_gen.flow_from_directory(test_path)

# ## Model # 1

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D

#https://stats.stackexchange.com/questions/148139/rules-for-selecting-convolutional-neural-network-hyperparameters
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape = new_image_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape = new_image_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape = new_image_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))

# Dropouts help reduce overfitting by randomly turning neurons off during training.Turn off 50% of neurons.

model.add(Dropout(0.5))

# Last layer, remember its binary so we use sigmoid

model.add(Dense(1))
model.add(Activation('sigmoid'))                   # last layer is signmoid

model.compile(loss='binary_crossentropy',          # Note this is another loss type
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',patience=2)

# help(image_gen.flow_from_directory)

batch_size = 16

train_image_gen = image_gen.flow_from_directory(train_path,
                                                target_size = new_image_shape[:2],
                                                color_mode='rgb',
                                                batch_size=batch_size,
                                                class_mode='binary')

test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size= new_image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',
                                               shuffle=False)

train_image_gen.class_indices

import warnings
warnings.filterwarnings('ignore')

epochs = 20
history = model.fit(train_image_gen,epochs=epochs,
                    validation_data=test_image_gen,
                    callbacks=[early_stop])

# from tensorflow.keras.models import load_model
# model.save('malaria_detector.h5')

# ### Evaluate

var = model.metrics_names

model.evaluate_generator(test_image_gen)

history_df = pd.DataFrame(model.history.history)
history_df[['loss','val_loss']].plot()

from tensorflow.keras.preprocessing import image


# ### Predictions

# https://datascience.stackexchange.com/questions/13894/how-to-get-predictions-with-predict-generator-on-streaming-test-data-in-keras
pred_probabilities = model.predict_generator(test_image_gen)

pred_probabilities

test_image_gen.classes

y_predictions = pred_probabilities > 0.5

# Numpy can treat this as True/False for us
y_predictions

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(test_image_gen.classes, y_predictions))

confusion_matrix(test_image_gen.classes, y_predictions)

infected_image = image.load_img(infected_cell_path,target_size = new_image_shape)



infected_image

type(infected_image)

infected_image = image.img_to_array(infected_image)

type(infected_image)

infected_image = np.expand_dims(infected_image, axis=0)

infected_image.shape

model.predict(infected_image)   # not infected is 0 else 1

train_image_gen.class_indices

test_image_gen.class_indices

# #### Reports

# #### Predictions go wrong!

# ### Conclusion : Final thoughts
# This model has the accuracy and validation for the ....

# ### Conclusion : This is the line from the pycharm

ImageDataGenerator


