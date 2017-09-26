import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

import Util
import LoadData
mnist = LoadData.read_data_sets( "MNIST_data/", one_hot=True )


