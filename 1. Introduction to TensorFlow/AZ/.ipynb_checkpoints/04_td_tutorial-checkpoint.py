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
#     display_name: PyCharm (TF_786)
#     language: python
#     name: pycharm-cf112a48
# ---

import tensorflow as tf

print('hello world')

  print(tf.__version__)

  # Create Tensor
  tensor1 = tf.range(5)
  
  #print(dir(tf.data.Dataset))
  #Create dataset, this will return object of TensorSliceDataset
  dataset = tf.data.Dataset.from_tensor_slices(tensor1)
  print(dataset)
  print("Original dataset")
  for i in dataset:    
      print(i)

  dataset = dataset.batch(batch_size=2)
  print("dataset after applying batch method")
  for i in dataset:
      print(i)
    
  print("\ndataset after applying take() method")
  for i in dataset.take(4):
        print(i)

# Create Tensor
tensor1 = tf.range(6)

#print(dir(tf.data.Dataset))
#Create dataset, this will return object of TensorSliceDataset
dataset = tf.data.Dataset.from_tensor_slices(tensor1)
print(dataset)
print("Original dataset")
for i in dataset:    
    print(i)


#Using batch method with repeat
dataset = dataset.repeat(3).batch(batch_size=2)
print("\ndataset after applying batch method with repeat()")
for i in dataset:
    print(i)

print("dataset after applying take() method")
for i in dataset.take(4):
  print(i)

# Transforming dataset items using map()
print("dataset after applying map function")
dataset = dataset.map(lambda x : x*x*x)
for i in dataset:
    print(i)

# note 8, 27, etc

# Playing 52 deck cards
tensor1 = tf.range(52)
dataset = tf.data.Dataset.from_tensor_slices(tensor1)
print(dataset)
print("Original dataset")
#for i in dataset:    
    #print(i)
    
#Using batch method with repeat
dataset = dataset.shuffle(52).batch(4)

for i in dataset:    
    print(i)


# Playing Ludo
tensor1 = tf.range(6)
dataset = tf.data.Dataset.from_tensor_slices(tensor1)
print(dataset)
print("Original dataset")
#for i in dataset:    
    #print(i)
    
#Using batch method with repeat
dataset = dataset.shuffle(1)

for i in dataset:    
    print(i)

print("----")
dataset = dataset.shuffle(2)

for i in dataset:    
    print(i)    

print("----")
dataset = dataset.shuffle(3)

for i in dataset:    
    print(i)  
