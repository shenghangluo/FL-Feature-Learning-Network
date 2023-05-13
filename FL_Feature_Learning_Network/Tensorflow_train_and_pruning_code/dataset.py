import numpy as np
from scipy.io import loadmat

N = 16
time_indice = 251
data_train = './Prossed_X_Dataset_16QAM_1000_PreCDC_5WDM32G_train_2.mat'
data_valid = './Prossed_X_Dataset_16QAM_1000_PreCDC_5WDM32G_valid_2.mat'

data_train_Y = './Prossed_Y_Dataset_16QAM_1000_PreCDC_5WDM32G_train_2.mat'
data_valid_Y = './Prossed_Y_Dataset_16QAM_1000_PreCDC_5WDM32G_valid_2.mat'

class Dataset(object):
    def __init__(self, val_size):
        self._val_size = val_size

    def load_data(self):

        # Load the Dataset
        A = loadmat(data_train)['Input_Symbol_real_X']
        A = np.array(A)
        Input_Symbol_Real = np.reshape(A, [-1, 1])
        print("Input_Symbol_real_X", Input_Symbol_Real.shape)
        A = loadmat(data_train)['Input_Symbol_image_X']
        A = np.array(A)
        Input_Symbol_Image = np.reshape(A, [-1, 1])

        A = loadmat(data_train_Y)['Input_Symbol_real_Y']
        A = np.array(A)
        Input_Symbol_Y_Real = np.reshape(A, [-1, 1])
        print("Input_Symbol_real_X", Input_Symbol_Real.shape)
        A = loadmat(data_train_Y)['Input_Symbol_image_Y']
        A = np.array(A)
        Input_Symbol_Y_Image = np.reshape(A, [-1, 1])

        A = loadmat(data_train)['X_real']
        A = np.array(A)
        X_real = np.reshape(A, [-1, time_indice, 1])
        print("X_real", X_real.shape)
        A = loadmat(data_train)['X_image']
        A = np.array(A)
        X_image = np.reshape(A, [-1, time_indice, 1])
        XX_Training = np.c_[X_real, X_image]

        # Y pol
        A = loadmat(data_train_Y)['Y_real']
        A = np.array(A)
        Y_real = np.reshape(A, [-1, time_indice, 1])
        A = loadmat(data_train_Y)['Y_image']
        A = np.array(A)
        Y_image = np.reshape(A, [-1, time_indice, 1])
        Y_Training = np.c_[Y_real, Y_image]
        X_Training = np.c_[XX_Training, Y_Training]

        A = loadmat(data_train)['Transmit_real_X']
        A = np.array(A)
        Transmit_Symbol_Real = np.reshape(A, [-1, 1])
        A = loadmat(data_train)['Transmit_image_X']
        A = np.array(A)
        Transmit_Symbol_Image = np.reshape(A, [-1, 1])
 
        A = loadmat(data_train_Y)['Transmit_real_Y']
        A = np.array(A)
        Transmit_Symbol_Y_Real = np.reshape(A, [-1, 1])
        A = loadmat(data_train_Y)['Transmit_image_Y']
        A = np.array(A)
        Transmit_Symbol_Y_Image = np.reshape(A, [-1, 1])

        Input_Symbol_Complex = np.c_[Input_Symbol_Real,Input_Symbol_Image]
        Transmit_Symbol_Complex = np.c_[Transmit_Symbol_Real,Transmit_Symbol_Image]

        Input_Symbol_Y_Complex = np.c_[Input_Symbol_Y_Real,Input_Symbol_Y_Image]
        Transmit_Symbol_Y_Complex = np.c_[Transmit_Symbol_Y_Real,Transmit_Symbol_Y_Image]

        # Valid
        A = loadmat(data_valid)['Input_Symbol_real_X']
        A = np.array(A)
        Input_Symbol_Real_valid = np.reshape(A, [-1, 1])
        A = loadmat(data_valid)['Input_Symbol_image_X']
        A = np.array(A)
        Input_Symbol_Image_valid = np.reshape(A, [-1, 1])

        A = loadmat(data_valid_Y)['Input_Symbol_real_Y']
        A = np.array(A)
        Input_Symbol_Real_Y_valid = np.reshape(A, [-1, 1])
        A = loadmat(data_valid_Y)['Input_Symbol_image_Y']
        A = np.array(A)
        Input_Symbol_Image_Y_valid = np.reshape(A, [-1, 1])

        A = loadmat(data_valid)['X_real']
        A = np.array(A)
        X_real_valid = np.reshape(A, [-1, time_indice, 1])
        A = loadmat(data_valid)['X_image']
        A = np.array(A)
        X_image_valid = np.reshape(A, [-1, time_indice, 1])
        XX_valid = np.c_[X_real_valid, X_image_valid]

        # Y pol
        A = loadmat(data_valid_Y)['Y_real']
        A = np.array(A)
        Y_real = np.reshape(A, [-1, time_indice, 1])
        A = loadmat(data_valid_Y)['Y_image']
        A = np.array(A)
        Y_image = np.reshape(A, [-1, time_indice, 1])
        Y_valid = np.c_[Y_real, Y_image]
        X_valid = np.c_[XX_valid, Y_valid]

        A = loadmat(data_valid)['Transmit_real_X']
        A = np.array(A)
        Transmit_Symbol_Real_valid = np.reshape(A, [-1, 1])
        A = loadmat(data_valid)['Transmit_image_X']
        A = np.array(A)
        Transmit_Symbol_Image_valid = np.reshape(A, [-1, 1])

        A = loadmat(data_valid_Y)['Transmit_real_Y']
        A = np.array(A)
        Transmit_Symbol_Real_Y_valid = np.reshape(A, [-1, 1])
        A = loadmat(data_valid_Y)['Transmit_image_Y']
        A = np.array(A)
        Transmit_Symbol_Image_Y_valid = np.reshape(A, [-1, 1])

        Input_Symbol_Complex_valid = np.c_[Input_Symbol_Real_valid, Input_Symbol_Image_valid]
        Transmit_Symbol_Complex_valid = np.c_[Transmit_Symbol_Real_valid, Transmit_Symbol_Image_valid]

        Input_Symbol_Complex_Y_valid = np.c_[Input_Symbol_Real_Y_valid, Input_Symbol_Image_Y_valid]
        Transmit_Symbol_Complex_Y_valid = np.c_[Transmit_Symbol_Real_Y_valid, Transmit_Symbol_Image_Y_valid]


        return {'X': X_Training,
                'Rx': Input_Symbol_Complex,
                'Rx_Y': Input_Symbol_Y_Complex,
                'Tx_Y': Transmit_Symbol_Y_Complex,
                'Tx': Transmit_Symbol_Complex,

                'X_valid': X_valid,
                'Rx_valid': Input_Symbol_Complex_valid,
                'Rx_Y_valid': Input_Symbol_Complex_Y_valid,
                'Tx_Y_valid': Transmit_Symbol_Complex_Y_valid,
                'Tx_valid': Transmit_Symbol_Complex_valid
                }

    @staticmethod
    def shuffle_batch(X, Rx,  Rx_Y, Tx_Y, Tx, batch_size):
        rnd_idx = np.random.permutation(len(X))
        n_batches = len(X) // batch_size
        for batch_idx in np.array_split(rnd_idx, n_batches):
            X_batch, Rx_batch, Rx_Y_batch, Tx_Y_batch, Tx_batch = X[batch_idx], Rx[batch_idx], Rx_Y[batch_idx], Tx_Y[batch_idx], Tx[batch_idx]
            yield X_batch, Rx_batch, Rx_Y_batch, Tx_Y_batch, Tx_batch
