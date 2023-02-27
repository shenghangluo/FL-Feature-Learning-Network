import tensorflow as tf
import numpy as np
from scipy.io import loadmat

M = 2      # number of output symbol
time_indice = 65560
Nstate = 65536
hidden_unit = 25
def complex_multiply(x, y):
    xr = x[:, 0]
    xi = x[:, 1]
    yr = y[:, 0]
    yi = y[:, 1]
    return tf.stack([xr * yr - xi * yi, xr * yi + xi * yr], axis=1)

class Model(object):
    def __init__(self):
        # The FO network
        self.X = tf.compat.v1.placeholder(tf.float32, shape=(None, time_indice, 4), name="X")
        self.Rx = tf.compat.v1.placeholder(tf.float32, shape=(None, M), name="Rx")
        self.Rx_Y = tf.compat.v1.placeholder(tf.float32, shape=(None, M), name="Rx_Y")
        self.Tx_Y = tf.compat.v1.placeholder(tf.float32, shape=(None, M), name="Tx_Y") 
        self.Tx = tf.compat.v1.placeholder(tf.float32, shape=(None, M), name="Tx")        
        self.rate = tf.compat.v1.placeholder(tf.float32, shape=(), name="Rate")
        self.state = tf.compat.v1.placeholder(tf.float32, shape=(Nstate, 25 * hidden_unit), name="state")

        self.cell_fw = tf.contrib.rnn.GRUCell(num_units=hidden_unit)
        self.cell_bw = tf.contrib.rnn.GRUCell(num_units=hidden_unit)

        (output_fw, output_bw), _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, self.X,
                                                                              dtype=tf.float32)
        self.X_RNN_1 = output_fw[:, 113:138, :] + output_bw[:, 113:138,:]
        self.X_ADD = tf.reshape(self.X_RNN_1, [-1, 25*hidden_unit])
        self.dense_1 = tf.keras.layers.Dense(2, activation=None, name="hidden_1")
        self.X_1 = self.dense_1(self.state)
        self.X_1 = tf.nn.leaky_relu(self.X_1, alpha=0.5, name=None)

        self.dense_2 = tf.keras.layers.Dense(10, activation=None, name="hidden_2")
        self.X_1 = self.dense_2(self.X_1)
        self.X_2 = tf.nn.leaky_relu(self.X_1, alpha=0.5, name=None)
        self.X_2_drop = tf.nn.dropout(self.X_2, rate=self.rate)
        self.dense_3 = tf.keras.layers.Dense(2, activation=None, name="out")
        self.AD_NL = self.dense_3(self.X_2_drop)
        self.test_point = self.AD_NL

        self.out_layer = self.Rx-self.AD_NL
        self.out_layer_Y = self.out_layer
    def get_output(self):
        return self.out_layer
    def get_X_ADD(self):
        return self.test_point

