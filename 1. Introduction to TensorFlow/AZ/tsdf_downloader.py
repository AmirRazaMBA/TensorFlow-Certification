import tensorflow as tf
import tensorflow_datasets as tfds

tfds.list_builders()

data_dir = 'D:\\Sandbox\\Github\\DATA_TFDS'

name = 'sentiment140'

dataset, info = tfds.load(name=name,
                          # split=['train', 'test'],
                          data_dir=data_dir,
                          with_info=True,
                          as_supervised=True,  # mutually exclusive with split
                          shuffle_files=True,
                          download=True)
