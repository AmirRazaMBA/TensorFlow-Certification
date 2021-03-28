# ---
# jupyter:
#   jupytext:
#     formats: ipynb,md,py:nomarker
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

# ## CNN - Data Augmentation - Reduce over fitting

from google.colab import drive

drive.mount('/content/drive', force_remount=True)

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

# #### Retreive Data - OS and Filesystem

URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_dir = tf.keras.utils.get_file('cats_and_dogs_filterted.zip', origin=URL, extract=True)
base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print('total training cat images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)

print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)


# #### Helping Functions

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


# #### Batch and Image Parameters

BATCH_SIZE = 100
IMG_SHAPE = 150  # Our training data consists of images with width of 150 pixels and height of 150 pixels

# #### Augment Images

from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_gen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)

train_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_SHAPE, IMG_SHAPE))

image_gen_train = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_SHAPE, IMG_SHAPE),
                                                     class_mode='binary')

# #### Plot Augmented Images

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

image_gen_val = ImageDataGenerator(rescale=1. / 255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=BATCH_SIZE,
                                                 directory=validation_dir,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 class_mode='binary')

# #### Create Model

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.5),  # 50% of the values will be set to zero. This helps to prevent overfitting
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# #### Fit Model

epochs = 100
history = model.fit(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
)

# #!mkdir -p colab_model


model_filename = "colab_model/04_cat_dogs_aug_model"

model.save(model_filename)

# !cp -r colab_model/04_cat_dogs_aug_model /content/drive/MyDrive/ColabModels/

# #### Evaluate Model

pd.DataFrame(model.history.history).head()
pd.DataFrame(model.history.history).plot()
losses = pd.DataFrame(model.history.history)
losses[['loss', 'val_loss']].plot()
losses[['accuracy', 'val_accuracy']].plot()

# #### Reload Saved Model

filename = "04_cat_dogs_aug_model"
saved_model_filename = "G:\\My Drive\\ColabModels\\" + filename
saved_model_filename

new_model = tf.keras.models.load_model(saved_model_filename)
new_model.summary()

loss, acc = new_model.evaluate(val_data_gen, verbose=2)
print(f'Restored model, accuracy: {100 * acc}')


# #### Find Class Labels

# optional
def find_class_labels(data_gen):
    data_gen.class_indices
    dataset_labels = sorted(data_gen.class_indices.items(),
                            key=lambda pair: pair[1])
    dataset_labels = np.array([key.title() for key, value in dataset_labels])
    return dataset_labels


# any one will work

dataset_labels = find_class_labels(val_data_gen)

dataset_labels = find_class_labels(train_data_gen)

dataset_labels = ['Cats', 'Dogs']

# #### Prediction on validation batch

# Get data as a batch - batch has image and label size 100
# val_image_batch, val_label_batch = next(iter(val_data_gen))

iterator = iter(val_data_gen)
val_image_batch, val_label_batch = next(iterator)  # iterator.next() 

print("Validation batch shape:", val_image_batch.shape)

# batch label : cat or dog
# val_label_batch

# ##### Predict

model_predictions = new_model.predict(val_image_batch)

pred_df = pd.DataFrame(model_predictions)
pred_df.columns = dataset_labels
print("Prediction results for the first elements")
pred_df.head()

# #### Misclassified Examples

predicted_label_ids = np.argmax(model_predictions, axis=-1)

n = 11
print(int(val_label_batch[n]))
print(int(predicted_label_ids[n]))

plt.figure(figsize=(20, 15))
plt.subplots_adjust(hspace=0.5)
for n in range(20):
    if (int(predicted_label_ids[n]) == int(val_label_batch[n])):
        color = 'green'
    else:
        color = 'red'
    plt.subplot(6, 5, n + 1)
    plt.imshow(val_image_batch[n])
    plt.title(f"{dataset_labels[predicted_label_ids[n]]}".title(), color=color)
    # plt.title(f"True : {val_label_batch[n]} Predicted :{predicted_label_ids[n]}".title())
    plt.axis('off')
    _ = plt.suptitle("Model predictions (green: correct, red: incorrect)")

# #### Reports

from sklearn.metrics import classification_report, confusion_matrix

y_prediction = new_model.predict(val_data_gen).argmax(axis=1)

len(y_prediction)
y_prediction.shape
y_prediction_df = pd.DataFrame(y_prediction)
y_prediction_df
