import LoadData
mnist = LoadData.read_data_sets( 'MNIST_data/', one_hot = True )

import numpy as np

import tensorflow as tf
sess = tf.InteractiveSession()

