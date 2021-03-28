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
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# tfds.list_builders()

data_dir = 'D:\\Sandbox\\Github\\DATA_TFDS'

dataset, info = tfds.load(name="cifar10",
                          data_dir=data_dir,
                          with_info=True,
                          as_supervised=True,  # mutually exclusive with split
                          shuffle_files=True,
                          download=False)

print(info.features["label"].names)
print(info.features["label"].int2str(7))

train_dataset, test_dataset = dataset['train'], dataset['test']

tfds.as_dataframe(train_dataset.take(5), info)


def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


# ### Prepare data for Model

BATCH_SIZE = 64
TRAIN_SIZE = len(train_dataset)  # memory max = 1000
# if repeat() then model needs epoch/step
# train_dataset = #train_dataset.cache().repeat().shuffle(TRAIN_SIZE).batch(BATCH_SIZE).prefetch(TRAIN_SIZE)

train_dataset = train_dataset.map(normalize).cache().shuffle(TRAIN_SIZE).batch(BATCH_SIZE).prefetch(
    tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.map(normalize).batch(BATCH_SIZE).cache().prefetch(tf.data.experimental.AUTOTUNE)

# ### Prepare data for Prediction analysis

# data is batch sized
for test_images, test_labels in test_dataset.take(1):
    x_test = test_images.numpy()
    y_test = test_labels.numpy()

n = 7
plt.imshow(x_test[n])
print(info.features["label"].int2str(y_test[n]))
plt.show()





dataset, info = tfds.load(name="titanic",
                          data_dir=data_dir,
                          with_info=True,
                          as_supervised=True,  # mutually exclusive with split
                          shuffle_files=True,
                          download=False)
print(info)
print(info.features["label"].names)
print(info.features["label"].int2str(7))

train_dataset = dataset['train']

tfds.as_dataframe(train_dataset.take(5), info)

for example in train_dataset.take(1):
    print(" Input Features")
    x_test = example.numpy()

# Construct a tf.data.Dataset
ds = tfds.load('titanic', split='train', shuffle_files=True)

ds = ds.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
for example in ds.take(1):
    # Print features for the batch input
    print(" Input Features")
    print(example['features'])

    # Print labels for batch input
    print(" Input Labels")
    print(example['survived'])