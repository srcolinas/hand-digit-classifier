"""Model architectures.

All models defined in this module expect an image and output unnormalized
logits of shape [N, num_labels].

"""
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras import layers

class Baseline(Model):
    def __init__(self, num_labels):
        super(Baseline, self).__init__()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(100, activation='elu')
        self.dense2 = layers.Dense(num_labels)
        self.sofmax = layers.Softmax()

    def call(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.sofmax(x) 
        

class BasicCNN(Model):
    def __init__(self, conv_specs, dense_specs):
        super(BasicCNN, self).__init__()
        raise NotImplementedError

    def call(self, input, training=False):
        raise NotImplementedError