import numpy as np
from scipy.io import loadmat

N = 16
time_indice = 65560
data_test = './Prossed_long_sequence_Dataset_16QAM_1000_PreCDC_5WDM32G_X_test_2.mat'
data_test_Y = './Prossed_long_sequence_Dataset_16QAM_1000_PreCDC_5WDM32G_Y_test_2.mat'

state_path = './Processed_state.mat'

class DatasetMNIST(object):
    def __init__(self, val_size):
        self._val_size = val_size

    def load_data(self):
        # Load the Dataset
        # State
        A = loadmat(state_path)['state']
        A = np.array(A)
        state = A
        print(state.shape)
        # Testing
        A = loadmat(data_test)['Input_Symbol_real_X']
        A = np.array(A)
        Input_Symbol_Real_test = np.reshape(A, [-1, 1])
        A = loadmat(data_test)['Input_Symbol_image_X']
        A = np.array(A)
        Input_Symbol_Image_test = np.reshape(A, [-1, 1])

        A = loadmat(data_test_Y)['Input_Symbol_real_Y']
        A = np.array(A)
        Input_Symbol_Real_Y_test = np.reshape(A, [-1, 1])
        A = loadmat(data_test_Y)['Input_Symbol_image_Y']
        A = np.array(A)
        Input_Symbol_Image_Y_test = np.reshape(A, [-1, 1])

        A = loadmat(data_test)['X_real']
        A = np.array(A)
        X_real_test = np.reshape(A, [-1, time_indice, 1])
        A = loadmat(data_test)['X_image']
        A = np.array(A)
        X_image_test = np.reshape(A, [-1, time_indice, 1])
        XX_test = np.c_[X_real_test, X_image_test]

        # Y pol
        A = loadmat(data_test_Y)['Y_real']
        A = np.array(A)
        Y_real = np.reshape(A, [-1, time_indice, 1])
        A = loadmat(data_test_Y)['Y_image']
        A = np.array(A)
        Y_image = np.reshape(A, [-1, time_indice, 1])
        Y_Test = np.c_[Y_real, Y_image]

        X_test = np.c_[XX_test, Y_Test]

        A = loadmat(data_test)['Transmit_real_X']
        A = np.array(A)
        Transmit_Symbol_Real_test = np.reshape(A, [-1, 1])
        A = loadmat(data_test)['Transmit_image_X']
        A = np.array(A)
        Transmit_Symbol_Image_test = np.reshape(A, [-1, 1])

        A = loadmat(data_test_Y)['Transmit_real_Y']
        A = np.array(A)
        Transmit_Symbol_Real_Y_test = np.reshape(A, [-1, 1])
        A = loadmat(data_test_Y)['Transmit_image_Y']
        A = np.array(A)
        Transmit_Symbol_Image_Y_test = np.reshape(A, [-1, 1])

        Input_Symbol_Complex_test = np.c_[Input_Symbol_Real_test, Input_Symbol_Image_test]
        Transmit_Symbol_Complex_test = np.c_[Transmit_Symbol_Real_test, Transmit_Symbol_Image_test]

        Input_Symbol_Complex_Y_test = np.c_[Input_Symbol_Real_Y_test, Input_Symbol_Image_Y_test]
        Transmit_Symbol_Complex_Y_test = np.c_[Transmit_Symbol_Real_Y_test, Transmit_Symbol_Image_Y_test]

        return {'X': X_test,
                'Rx': Input_Symbol_Complex_test,
                'Rx_Y': Input_Symbol_Complex_Y_test,
                'Tx_Y': Transmit_Symbol_Complex_Y_test,
                'Tx': Transmit_Symbol_Complex_test,

                'X_test': X_test,
                'Rx_test': Input_Symbol_Complex_test,
                'Rx_Y_test': Input_Symbol_Complex_Y_test,
                'Tx_Y_test': Transmit_Symbol_Complex_Y_test,
                'Tx_test': Transmit_Symbol_Complex_test,
                'state': state
                }

    @staticmethod
    def shuffle_batch(X, Rx,  Rx_Y, Tx_Y, Tx, batch_size):
        rnd_idx = np.random.permutation(len(X))
        n_batches = len(X) // batch_size
        for batch_idx in np.array_split(rnd_idx, n_batches):
            X_batch, Rx_batch, Rx_Y_batch, Tx_Y_batch, Tx_batch = X[batch_idx], Rx[batch_idx], Rx_Y[batch_idx], Tx_Y[batch_idx], Tx[batch_idx]
            yield X_batch, Rx_batch, Rx_Y_batch, Tx_Y_batch, Tx_batch



    @staticmethod
    def shuffle_batch_SO(X, X_SO, Z, batch_size):
        rnd_idx = np.random.permutation(len(X))
        n_batches = len(X) // batch_size
        for batch_idx in np.array_split(rnd_idx, n_batches):
            X_batch, X_SO_batch, Z_batch = X[batch_idx], X_SO[batch_idx], Z[batch_idx]
            yield X_batch, X_SO_batch, Z_batch