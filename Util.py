import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from LoadData import DataUtils

sess = tf.InteractiveSession()

def main():
    trainFile_x = 'D:/PycharmProjects/MNIST/MNIST_data/train-images.idx3-ubyte'
    trainFile_y = 'D:/PycharmProjects/MNIST/MNIST_data/train-labels.idx1-ubyte'
    testFile_x = 'D:/PycharmProjects/MNIST/MNIST_data/t10k-images.idx3-ubyte'
    testFile_y = 'D:/PycharmProjects/MNIST/MNIST_data/t10k-labels.idx1-ubyte'

    train_X = DataUtils( fileName = trainFile_x ).getImage()
    train_Y = DataUtils( fileName = trainFile_y ).getLabel()
    test_X = DataUtils( testFile_x ).getImage()
    test_Y = DataUtils( testFile_y ).getLabel()

    return train_X, train_Y, test_X, test_Y

def data_test():
    # Loading the dataset
    train_X, train_Y, test_X, test_Y = main()
    print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)

    index = 0
    image = train_X[index]
    print(image.shape)

    image = image.reshape(28, -1)
    print(image.shape)

    plt.imshow(image)
    plt.show()
    print("Y = " + str(np.squeeze(train_Y[index,])))

def fit_data():
    train_X, train_Y, test_X, test_Y = main()
    train_X, test_X, = train_X.T,test_X.T
    train_Y = one_hot_matrix( train_Y, 10 )
    test_Y = one_hot_matrix( test_Y, 10 )
    return train_X, train_Y, test_X, test_Y

def one_hot_matrix( labels, C ):

    C = tf.constant( C, name = "C" )
    one_hot_matrix = tf.one_hot( labels, C, axis = 0 )
    one_hot = sess.run( one_hot_matrix )
    return one_hot

def create_placeholeers( n_x, n_y ):
    X = tf.placeholder( tf.float32, [n_x, None] )
    Y = tf.placeholder( tf.float32, [n_y, None] )
    return X, Y




