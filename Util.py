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
    train_X, test_X, = train_X.T, test_X.T
    train_Y = one_hot_matrix( train_Y, 10 )
    test_Y = one_hot_matrix( test_Y, 10 )
    print ( train_X.shape, train_Y.shape, test_X.shape, test_Y.shape )
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

def initialize_parameters():
    tf.set_random_seed( 1 )
    W1 = tf.get_variable( "W1", [30, 784], initializer = tf.contrib.layers.xavier_initializer( seed = 1 ) )
    b1 = tf.get_variable( "b1", [30, 1], initializer = tf.zeros_initializer() )
    W2 = tf.get_variable( "W2", [15, 30], initializer = tf.contrib.layers.xavier_initializer( seed = 1 ) )
    b2 = tf.get_variable( "b2", [15, 1], initializer = tf.zeros_initializer() )
    W3 = tf.get_variable( "W3", [7, 15], initializer = tf.contrib.layers.xavier_initializer( seed = 1 ) )
    b3 = tf.get_variable( "b3", [7, 1], initializer = tf.zeros_initializer() )

    parameters = {"W1" : W1,
                  "b1" : b1,
                  "W2" : W2,
                  "b2" : b2,
                  "W3" : W3,
                  "b3" : b3}
    return parameters

def forward_propagation( X, parameters ):

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.matmul( W1, X ) + b1
    A1 = tf.nn.relu( Z1 )
    Z2 = tf.matmul( W2, A1 ) + b2
    A2 = tf.nn.relu( Z2 )
    Z3 = tf.matmul( W3, A2 ) + b3

    return Z3

def compute_cost( Z3, Y ):
    logits = tf.transpose( Z3 )
    labels = tf.transpose( Y )

    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( logits = logits, labels = labels ) )

    return cost

def model( train_X, train_Y, test_X, test_Y, learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 32, print_cost = True ):
    ops.rest_default_graph()
    ( n_x, m ) = train_X.shape
    n_y = train_Y.shape[0]
    costs = []

    X, Y = create_placeholeers( n_x, n_y )

    parameters = initialize_parameters()

    Z3 = forward_propagation( X, parameters )

    cost = compute_cost( Z3, Y )

    optimizer = tf.train.AdamOptimizer( learning_rate = learning_rate ).minimize( cost )

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run( init )

        for epoch in range( num_epochs ):
            epoch_cost = 0
            num_minibatches = int( m / minibatch_size )
            seed = seed + 1
            minibatches = random_mini_batches( train_X, train_Y, minibatch_size, seed )

            for minibatch in minibatches:
                ( minibatch_X, minibatch_Y ) = minibatch

                _, minibatch_cost = sess.run( [optimizer, cost], feed_dict = {X : minibatch_X, Y : minibatch_Y})

                epoch_cost += minibatch_cost / num_minibatches   #全局成本行函数
            if print_cost == True and epoch % 100 == 0:
                print ( "Cost after epoch %i: %f" % ( epoch, epoch_cost ) )
            if print_cost == True and epoch % 5 == 0:
                costs.append( epoch_cost )


        plt.plot( np.squeeze*( costs ) )
        plt.ylabel( "cost" )
        plt.xlabel( "iteration ( per tens)" )
        plt.title( "Learning rate = " + str( learning_rate ) )
        plt.show()

        parameters = sess.run( parameters )
        print ( "Parameters have been trained" )

        correct_prediction = tf.equal( tf.argmax( Z3 ), tf.argmax( Y ) )

        accuracy = tf.reduce_mean( tf.cast( correct_prediction, "float" ) )

        print( "Train Accuracy:", accuracy.eval( {X : train_X, Y : train_Y} ) )
        print( "Test Accuracy:", accuracy.eval( {X : test_X, Y : test_Y} ) )

        return parameters



def random_mini_batches( X, Y, mini_batch_size, seed ):
    m = X.shape[1]
    mini_batches = []

    # Step 1: Shuffle ( X, Y )
    permutation = list( np.random.permutation( m ) )
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape( ( 1, m ) )

    # Step 2: Partition ( shuffed_X, shuffed_Y ), Minus the end case
    num_complete_minibatches = math.floor( m / mini_batches_size )
    for k in range( 0, num_complete_minibatches ):
        mini_batch_X = shuffed_X[:, k * mini_batch_size : mini_batch_size * ( k + 1 )]
        mini_batch_Y = shuffed_Y[:, k * mini_batch_size : mini_batch_size * ( k + 1 )]
        mini_batch = ( mini_batch_X, mini_batch_Y )
        mini_batches.append( mini_batch )

    # Handing the end case ( last mini_batch < mini_batch_size )
    if m % mini_batch_size != 0:
        mini_batch_X = shuffed_X[:, mini_batch_size * num_complete_minibatches : m]
        mini_batch_Y = shuffed_Y[:, mini_batch_size * num_complete_minibatches : m]

        mini_batch = ( mini_batch_X, mini_batch_Y )
        mini_batches.append( mini_batch )

    return mini_batches
