import tensorflow as tf
from dataset import DatasetMNIST
from model import Model
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from scipy import special
from os.path import exists

learning_rate = 0.0001
pruning_iterations = 5
total_iterations = 0
pruning_schedule = (np.ceil(np.ceil(2.0 ** (-np.arange(pruning_iterations, 0, -1)) * total_iterations)))
epochs = np.zeros(pruning_iterations+1)
pruning_factors = np.array([[0.56, 0.56, 0.56, 0.56], [0.56, 0.56, 0.56, 0.56], [0.56, 0.56, 0.56, 0.56], [0.56, 0.56, 0.56, 0.56], [0.56, 0.56, 0.56, 0.56], [0.56, 0.56, 0.56, 0.56]])
for idx, prune_step in enumerate(pruning_schedule):
    if idx==0:
        epochs[idx] = prune_step
    else:
        epochs[idx] = prune_step - pruning_schedule[idx-1]
epochs[pruning_iterations] = total_iterations-pruning_schedule[pruning_iterations-1]
epochs = epochs.astype('int32')
print("epochs", epochs)

# A helper function for PCM_encode and elsewhere
def to_bin(data, width):
    data_str = bin(data & (2**width-1))[2:].zfill(width)
    return [int(x) for x in tuple(data_str)]
def MQAM_gray_decode(x_hat, M=4):
    # Inverse Gray code LUTs for 2, 4, 8, 16, 32, and 64 QAM
    # which employs M = 1, 2, 3, 4, 5, and 6  bits per symbol
    QAM2_Mapping = np.array([-1, 1])
    QAM4_Mapping = np.array([-0.707106781186548 + 0.707106781186548j, -0.707106781186548 - 0.707106781186548j,
                             0.707106781186548 + 0.707106781186548j, 0.707106781186548 - 0.707106781186548j])
    QAM8_Mapping = np.array([-1.22474487139159 + 0.408248290463863j, -1.22474487139159 - 0.408248290463863j,
                             -0.408248290463863 + 0.408248290463863j, -0.408248290463863 - 0.408248290463863j,
                             1.22474487139159 + 0.408248290463863j, 1.22474487139159 - 0.408248290463863j,
                             0.408248290463863 + 0.408248290463863j, 0.408248290463863 - 0.408248290463863j])
    QAM16_Mapping = np.array([-0.948683298050514 + 0.948683298050514j, -0.948683298050514 + 0.316227766016838j,
                              -0.948683298050514 - 0.948683298050514j, -0.948683298050514 - 0.316227766016838j,
                              -0.316227766016838 + 0.948683298050514j, -0.316227766016838 + 0.316227766016838j,
                              -0.316227766016838 - 0.948683298050514j, -0.316227766016838 - 0.316227766016838j,
                              0.948683298050514 + 0.948683298050514j, 0.948683298050514 + 0.316227766016838j,
                              0.948683298050514 - 0.948683298050514j, 0.948683298050514 - 0.316227766016838j,
                              0.316227766016838 + 0.948683298050514j, 0.316227766016838 + 0.316227766016838j,
                              0.316227766016838 - 0.948683298050514j, 0.316227766016838 - 0.316227766016838j])
    QAM32_Mapping = np.array([-0.670820393249937 + 1.11803398874990j, -0.223606797749979 + 1.11803398874990j,
                              -0.670820393249937 - 1.11803398874990j, -0.223606797749979 - 1.11803398874990j,
                              -1.11803398874990 + 0.670820393249937j, -1.11803398874990 + 0.223606797749979j,
                              -1.11803398874990 - 0.670820393249937j, -1.11803398874990 - 0.223606797749979j,
                              -0.223606797749979 + 0.670820393249937j, -0.223606797749979 + 0.223606797749979j,
                              -0.223606797749979 - 0.670820393249937j, -0.223606797749979 - 0.223606797749979j,
                              -0.670820393249937 + 0.670820393249937j, -0.670820393249937 + 0.223606797749979j,
                              -0.670820393249937 - 0.670820393249937j, -0.670820393249937 - 0.223606797749979j,
                              0.670820393249937 + 1.11803398874990j, 0.223606797749979 + 1.11803398874990j,
                              0.670820393249937 - 1.11803398874990j, 0.223606797749979 - 1.11803398874990j,
                              1.11803398874990 + 0.670820393249937j, 1.11803398874990 + 0.223606797749979j,
                              1.11803398874990 - 0.670820393249937j, 1.11803398874990 - 0.223606797749979j,
                              0.223606797749979 + 0.670820393249937j, 0.223606797749979 + 0.223606797749979j,
                              0.223606797749979 - 0.670820393249937j, 0.223606797749979 - 0.223606797749979j,
                              0.670820393249937 + 0.670820393249937j, 0.670820393249937 + 0.223606797749979j,
                              0.670820393249937 - 0.670820393249937j, 0.670820393249937 - 0.223606797749979j])
    QAM64_Mapping = np.array([-1.08012344973464 + 1.08012344973464j, -1.08012344973464 + 0.771516749810460j,
                              -1.08012344973464 + 0.154303349962092j, -1.08012344973464 + 0.462910049886276j,
                              -1.08012344973464 - 1.08012344973464j, -1.08012344973464 - 0.771516749810460j,
                              -1.08012344973464 - 0.154303349962092j, -1.08012344973464 - 0.462910049886276j,
                              -0.771516749810460 + 1.08012344973464j, -0.771516749810460 + 0.771516749810460j,
                              -0.771516749810460 + 0.154303349962092j, -0.771516749810460 + 0.462910049886276j,
                              -0.771516749810460 - 1.08012344973464j, -0.771516749810460 - 0.771516749810460j,
                              -0.771516749810460 - 0.154303349962092j, -0.771516749810460 - 0.462910049886276j,
                              -0.154303349962092 + 1.08012344973464j, -0.154303349962092 + 0.771516749810460j,
                              -0.154303349962092 + 0.154303349962092j, -0.154303349962092 + 0.462910049886276j,
                              -0.154303349962092 - 1.08012344973464j, -0.154303349962092 - 0.771516749810460j,
                              -0.154303349962092 - 0.154303349962092j, -0.154303349962092 - 0.462910049886276j,
                              -0.462910049886276 + 1.08012344973464j, -0.462910049886276 + 0.771516749810460j,
                              -0.462910049886276 + 0.154303349962092j, -0.462910049886276 + 0.462910049886276j,
                              -0.462910049886276 - 1.08012344973464j, -0.462910049886276 - 0.771516749810460j,
                              -0.462910049886276 - 0.154303349962092j, -0.462910049886276 - 0.462910049886276j,
                              1.08012344973464 + 1.08012344973464j, 1.08012344973464 + 0.771516749810460j,
                              1.08012344973464 + 0.154303349962092j, 1.08012344973464 + 0.462910049886276j,
                              1.08012344973464 - 1.08012344973464j, 1.08012344973464 - 0.771516749810460j,
                              1.08012344973464 - 0.154303349962092j, 1.08012344973464 - 0.462910049886276j,
                              0.771516749810460 + 1.08012344973464j, 0.771516749810460 + 0.771516749810460j,
                              0.771516749810460 + 0.154303349962092j, 0.771516749810460 + 0.462910049886276j,
                              0.771516749810460 - 1.08012344973464j, 0.771516749810460 - 0.771516749810460j,
                              0.771516749810460 - 0.154303349962092j, 0.771516749810460 - 0.462910049886276j,
                              0.154303349962092 + 1.08012344973464j, 0.154303349962092 + 0.771516749810460j,
                              0.154303349962092 + 0.154303349962092j, 0.154303349962092 + 0.462910049886276j,
                              0.154303349962092 - 1.08012344973464j, 0.154303349962092 - 0.771516749810460j,
                              0.154303349962092 - 0.154303349962092j, 0.154303349962092 - 0.462910049886276j,
                              0.462910049886276 + 1.08012344973464j, 0.462910049886276 + 0.771516749810460j,
                              0.462910049886276 + 0.154303349962092j, 0.462910049886276 + 0.462910049886276j,
                              0.462910049886276 - 1.08012344973464j, 0.462910049886276 - 0.771516749810460j,
                              0.462910049886276 - 0.154303349962092j, 0.462910049886276 - 0.462910049886276j])

    N_symb = len(x_hat)
    N_word = int(np.log2(M))

    data_hat = np.zeros(N_symb * N_word, dtype=int)
    # Create the serial bit stream using Gray decoding, msb to lsb
    for k in range(N_symb):
        if M == 2:  # special case for BPSK
            data_hat[k * N_word:(k + 1) * N_word] = to_bin((np.abs(QAM2_Mapping - x_hat[k])).argmin(), N_word)
        elif M == 4:  # total points of the square constellation
            data_hat[k * N_word:(k + 1) * N_word] = to_bin((np.abs(QAM4_Mapping - x_hat[k])).argmin(), N_word)
        elif M == 8:
            data_hat[k * N_word:(k + 1) * N_word] = to_bin((np.abs(QAM8_Mapping - x_hat[k])).argmin(), N_word)
        elif M == 16:
            data_hat[k * N_word:(k + 1) * N_word] = to_bin((np.abs(QAM16_Mapping - x_hat[k])).argmin(), N_word)
        elif M == 32:
            data_hat[k * N_word:(k + 1) * N_word] = to_bin((np.abs(QAM32_Mapping - x_hat[k])).argmin(), N_word)
        elif M == 64:
            data_hat[k * N_word:(k + 1) * N_word] = to_bin((np.abs(QAM64_Mapping - x_hat[k])).argmin(), N_word)
        else:
            raise ValueError('M must be 2, 4, 8, 16, 32 or 64')
    return data_hat

# Calculate the performance of NN model (BER and Q-Factor)
def evaluate_output(outputX, TransX, M):
    Rx_SymbolsX_Com = np.reshape(outputX[:,0]+1j*outputX[:,1], [1, outputX.shape[0]])
    Tx_SymbolsX_Com = np.reshape(TransX[:,0]+1j*TransX[:,1], [1, TransX.shape[0]])
    # QAM Demapper
    Rx_BitsX = np.zeros((Rx_SymbolsX_Com.shape[0], int(Rx_SymbolsX_Com.shape[1]*np.log2(M))))
    Tx_BitsX = np.zeros((Tx_SymbolsX_Com.shape[0], int(Tx_SymbolsX_Com.shape[1]*np.log2(M))))
    for i in range(Rx_SymbolsX_Com.shape[0]):
        Rx_BitsX[i, :] = MQAM_gray_decode(Rx_SymbolsX_Com[i, :], M=M)
        Tx_BitsX[i, :] = MQAM_gray_decode(Tx_SymbolsX_Com[i, :], M=M)
    # BER calculations
    Frame_ErrorsX = np.sum(Tx_BitsX[0, 100:Tx_BitsX.shape[1] - 100] != Rx_BitsX[0, 100:Rx_BitsX.shape[1] - 100])
    Total_Errors = Frame_ErrorsX
    Total_Bits = (Rx_BitsX.shape[1] - 200)
    BER_local = Total_Errors / Total_Bits
    Q_factor_dB_local = 20 * np.log10(np.sqrt(2) * (special.erfcinv(2 * BER_local)))
    print("For CoI: BER = %f, Q-Factor = %f dB" % (BER_local, Q_factor_dB_local))
    return BER_local, Q_factor_dB_local

def cosine_annealing(epoch, n_epochs, n_cycles, lrate_max):
	epochs_per_cycle = np.floor(n_epochs/n_cycles)
	cos_inner = (np.pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
	return lrate_max/2 * (np.cos(cos_inner) + 1)

def Wegith_initialization(model, weights):
    for idx, weight in enumerate(weights):
        # print(idx, x)
        if idx == 0:
            tf.keras.backend.set_value(model.cell_fw.weights[0], weight)
        if idx == 1:
            tf.keras.backend.set_value(model.cell_fw.weights[1], weight)
        if idx == 2:
            tf.keras.backend.set_value(model.cell_fw.weights[2], weight)
        if idx == 3:
            tf.keras.backend.set_value(model.cell_fw.weights[3], weight)
        if idx == 4:
            tf.keras.backend.set_value(model.cell_bw.weights[0], weight)
        if idx == 5:
            tf.keras.backend.set_value(model.cell_bw.weights[1], weight)
        if idx == 6:
            tf.keras.backend.set_value(model.cell_bw.weights[2], weight)
        if idx == 7:
            tf.keras.backend.set_value(model.cell_bw.weights[3], weight)
        if idx == 8:
            tf.keras.backend.set_value(model.dense_1.weights[0], weight)
        if idx == 9:
            tf.keras.backend.set_value(model.dense_1.weights[1], weight)
        if idx == 10:
            tf.keras.backend.set_value(model.dense_2.weights[0], weight)
        if idx == 11:
            tf.keras.backend.set_value(model.dense_2.weights[1], weight)
        if idx == 12:
            tf.keras.backend.set_value(model.dense_3.weights[0], weight)
        if idx == 13:
            tf.keras.backend.set_value(model.dense_3.weights[1], weight)


    return 0

def get_masks_for_model(sess, vars, pruning_factor):
    masks = []
    weight_matrices = []

    for var, val in zip(vars, vars):
        if 'gru_cell/gates/kernel' in val.name:
            pruning_weight = sess.run(val)
            masks.append(np.ones(pruning_weight.shape))
            weight_matrices.append(pruning_weight)
        if 'gru_cell/candidate/kernel' in val.name:
            pruning_weight = sess.run(val)
            masks.append(np.ones(pruning_weight.shape))
            weight_matrices.append(pruning_weight)

    if not isinstance(pruning_factor, np.ndarray):
        pruning_factor = np.ones(len(masks))*pruning_factor

    # That's it if no pruning is required
    if np.count_nonzero(pruning_factor) == 0:
        print("No Pruning, mask=1")
        return masks

    # Ideally, we should prune globally using norm of complex weights.
    # But here, we are going to prune each real weight matrix individually.
    for idx, weights in enumerate(weight_matrices):
        weights = np.abs(weights)
        threshold = np.percentile(np.abs(weights).ravel(), pruning_factor[idx] * 100)
        masks[idx][weights < threshold] = 0

    return masks

def Applying_mask(sess, vars, masks, model):
    # Apply Pruning Mask
    if len(masks)!=4:
        print("Wrong Mask!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    for var, val in zip(vars, vars):
        # print(val.name, val.shape)
        if 'fw/gru_cell/gates/kernel' in val.name:
            pruning_weight = np.array(sess.run(model.cell_fw.weights[0]))
            pruning_weight = pruning_weight * (np.ndarray.astype(masks[0], dtype=pruning_weight.dtype))
            tf.keras.backend.set_value(model.cell_fw.weights[0], pruning_weight)
        if '/fw/gru_cell/candidate/kernel' in val.name:
            pruning_weight = np.array(sess.run(model.cell_fw.weights[2]))
            pruning_weight = pruning_weight * (np.ndarray.astype(masks[1], dtype=pruning_weight.dtype))
            tf.keras.backend.set_value(model.cell_fw.weights[2], pruning_weight)
        if 'bw/gru_cell/gates/kernel' in val.name:
            pruning_weight = np.array(sess.run(model.cell_bw.weights[0]))
            pruning_weight = pruning_weight * (np.ndarray.astype(masks[2], dtype=pruning_weight.dtype))
            tf.keras.backend.set_value(model.cell_bw.weights[0], pruning_weight)
        if '/bw/gru_cell/candidate/kernel' in val.name:
            pruning_weight = np.array(sess.run(model.cell_bw.weights[2]))
            pruning_weight = pruning_weight * (np.ndarray.astype(masks[3], dtype=pruning_weight.dtype))
            tf.keras.backend.set_value(model.cell_bw.weights[2], pruning_weight)

def Mask_Gradient(gvs, vars, pruning_factors):
    masked_gvs = []
    idx = 0
    masks = {}
    for var, val in zip(vars, vars):
        if 'fw/gru_cell/gates/kernel' in val.name:
            k = tf.cast(tf.round(tf.size(val, out_type=tf.float32) * pruning_factors[idx]), dtype=tf.int32)
            w_reshaped = tf.reshape(val, [-1])
            _, indices = tf.nn.top_k(tf.negative(tf.abs(w_reshaped)), k, sorted=True, name=None)
            mask_1 = tf.scatter_nd_update(
                tf.Variable(tf.ones_like(w_reshaped, dtype=tf.float32), name="mask_gra_1", trainable=False),
                tf.reshape(indices, [-1, 1]), tf.zeros([k], tf.float32))
            masks[idx] = tf.reshape(mask_1, tf.shape(val))
            idx = idx+1
        if '/fw/gru_cell/candidate/kernel' in val.name:
            k = tf.cast(tf.round(tf.size(val, out_type=tf.float32) * pruning_factors[idx]), dtype=tf.int32)
            w_reshaped = tf.reshape(val, [-1])
            _, indices = tf.nn.top_k(tf.negative(tf.abs(w_reshaped)), k, sorted=True, name=None)
            mask_1 = tf.scatter_nd_update(
                tf.Variable(tf.ones_like(w_reshaped, dtype=tf.float32), name="mask_gra_2", trainable=False),
                tf.reshape(indices, [-1, 1]), tf.zeros([k], tf.float32))
            masks[idx] = tf.reshape(mask_1, tf.shape(val))
            idx = idx+1
        if 'bw/gru_cell/gates/kernel' in val.name:
            k = tf.cast(tf.round(tf.size(val, out_type=tf.float32) * pruning_factors[idx]), dtype=tf.int32)
            w_reshaped = tf.reshape(val, [-1])
            _, indices = tf.nn.top_k(tf.negative(tf.abs(w_reshaped)), k, sorted=True, name=None)
            mask_1 = tf.scatter_nd_update(
                tf.Variable(tf.ones_like(w_reshaped, dtype=tf.float32), name="mask_gra_3", trainable=False),
                tf.reshape(indices, [-1, 1]), tf.zeros([k], tf.float32))
            masks[idx] = tf.reshape(mask_1, tf.shape(val))
            idx = idx+1
        if '/bw/gru_cell/candidate/kernel' in val.name:
            k = tf.cast(tf.round(tf.size(val, out_type=tf.float32) * pruning_factors[idx]), dtype=tf.int32)
            w_reshaped = tf.reshape(val, [-1])
            _, indices = tf.nn.top_k(tf.negative(tf.abs(w_reshaped)), k, sorted=True, name=None)
            mask_1 = tf.scatter_nd_update(
                tf.Variable(tf.ones_like(w_reshaped, dtype=tf.float32), name="mask_gra_4", trainable=False),
                tf.reshape(indices, [-1, 1]), tf.zeros([k], tf.float32))
            masks[idx] = tf.reshape(mask_1, tf.shape(val))
            idx = idx+1

    idx = 0
    for grad, var in gvs:
        masked_gvs.append([grad, var])
        if "fw/while/gru_cell/MatMul/" in grad.name:
            masked_grad = tf.math.multiply(grad, masks[idx], name='final_grad_mask_1')
            masked_gvs.append([masked_grad, var])
            idx = idx + 1
        if "fw/while/gru_cell/MatMul_1/" in grad.name:
            masked_grad = tf.math.multiply(grad, masks[idx], name='final_grad_mask_2')
            masked_gvs.append([masked_grad, var])
            idx = idx + 1
        if "bw/while/gru_cell/MatMul/" in grad.name:
            masked_grad = tf.math.multiply(grad, masks[idx], name='final_grad_mask_3')
            masked_gvs.append([masked_grad, var])
            idx = idx + 1
        if "bw/while/gru_cell/MatMul_1/" in grad.name:
            masked_grad = tf.math.multiply(grad, masks[idx], name='final_grad_mask_4')
            masked_gvs.append([masked_grad, var])
            idx = idx + 1

    return masked_gvs

class Trainer(object):
    def __init__(self, n_epochs, tr_batch_size, optimizer_params, Train_Test):
        self._learning_rate = tf.placeholder(tf.float32, shape=[], name="learn_rate")
        self._n_epochs = n_epochs
        self._batch_size = tr_batch_size
        self._dataset = DatasetMNIST(val_size=10000)
        self._model = Model()
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
        self._writer_1 = tf.summary.FileWriter('./summary1')
        self._writer_2 = tf.summary.FileWriter('./summary2')
        self._Train_Test = Train_Test
        self._vars = tf.trainable_variables()
        self._pruning_factor = tf.placeholder(tf.float32, shape=[4], name="pruning_factor")

    def train_model(self):
        data = self._dataset.load_data()
        # loss function
        loss = tf.reduce_sum(tf.reduce_mean(tf.square(self._model.Tx-self._model.out_layer), axis=0))
        gvs = self._optimizer.compute_gradients(loss, self._vars)
        masked_gvs = Mask_Gradient(gvs, self._vars, self._pruning_factor)
        training_op = self._optimizer.apply_gradients(masked_gvs)
        saver = tf.train.Saver()
        write_op=tf.summary.merge_all()

        load_weight = 0
        # Load Trained Weights
        if load_weight == 1:
            self._weights = []
            idx = 0
            while(exists("./weight_"+str(idx)+".csv")):
                weight = np.loadtxt(open("./weight_"+str(idx)+".csv", "rb"), delimiter=",", skiprows=0)
                self._weights.append(weight)
                idx = idx + 1

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            init.run()
            writer = tf.summary.FileWriter('./graphs', sess.graph)
            flag=0
            training_loss_set = np.empty((0))
            testing_loss_set = np.empty((0))
            Q_set = np.empty((0))
            Q_set_Y = np.empty((0))
            if self._Train_Test == 0:
              saver.restore(sess, "./modelPrunedALL.ckpt")
              print("First-order Model Restored.")
              print("First-order Training Start")
              iterations = 0
              step_size = 2048
              base_lr = 0.0001
              max_lr = 0.0001
              total_epoch = 0

              Rx_Y_test = self._model.Rx_Y.eval(feed_dict={self._model.rate: 0.0, self._pruning_factor:pruning_factors[0], self._learning_rate: 0.0, self._model.X: data['X_test'], self._model.Rx: data['Rx_test'], self._model.Rx_Y: data['Rx_Y_test'], self._model.Tx_Y: data['Tx_Y_test'], self._model.Tx: data['Tx_test']})
              Tx_Y_test = self._model.Tx_Y.eval(feed_dict={self._model.rate: 0.0, self._pruning_factor:pruning_factors[0], self._learning_rate: 0.0, self._model.X: data['X_test'], self._model.Rx: data['Rx_test'], self._model.Rx_Y: data['Rx_Y_test'], self._model.Tx_Y: data['Tx_Y_test'], self._model.Tx: data['Tx_test']})
              _, Q_Y = evaluate_output(Rx_Y_test, Tx_Y_test, M=16)
              Q_set_Y  = np.append(Q_set_Y, Q_Y)
              print("Q_set_Y:", Q_set_Y )

              # Wegith_initialization(self._model, self._weights)

              for pruning_iteration in range(pruning_iterations + 1):
                  pruning_factor = pruning_factors[pruning_iteration]
                  epoch_prune = epochs[pruning_iteration]
                  if flag == 1:
                      print("flag is turn on")
                      break
                  for epoch in range(0, epoch_prune):
                      ii = 0
                      train_loss = 0
                      valid_loss = 0
                      if flag == 1:
                          print("flag is turn on")
                          break

                      mask = get_masks_for_model(sess, self._vars, pruning_factor)

                      for X_batch, Rx_batch, Rx_Y_batch, Tx_Y_batch, Tx_batch in self._dataset.shuffle_batch(data['X'], data['Rx'], data['Rx_Y'], data['Tx_Y'], data['Tx'], self._batch_size):
                          cycle = np.floor(1 + iterations / (2 * step_size))
                          x = np.abs(iterations / step_size - 2 * cycle + 1)
                          learning_rate = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x))
                          Applying_mask(sess, self._vars, mask, self._model)

                          _, loss_batch = sess.run([training_op, loss], feed_dict={self._model.rate: 0.5, self._pruning_factor: pruning_factor, self._learning_rate: learning_rate, self._model.X: X_batch, self._model.Rx: Rx_batch, self._model.Rx_Y: Rx_Y_batch, self._model.Tx_Y: Tx_Y_batch, self._model.Tx: Tx_batch})
                          train_loss += loss_batch
                          iterations += 1
                          ii += 1
                      valid_loss = sess.run(loss, feed_dict={self._model.rate: 0.0, self._pruning_factor: pruning_factor,
                                                       self._learning_rate: learning_rate,
                                                       self._model.X: data['X_valid'],
                                                       self._model.Rx: data['Rx_valid'],
                                                       self._model.Rx_Y: data['Rx_Y_valid'],
                                                       self._model.Tx_Y: data['Tx_Y_valid'],
                                                       self._model.Tx: data['Tx_valid']})
                      print(total_epoch + epoch + 1, "Training Loss:", train_loss, "Validation Loss", valid_loss)
                      training_loss_set = np.append(training_loss_set, train_loss / ii)
                      valid_loss_set = np.append(valid_loss_set, valid_loss)
                      Rx_valid = self._model.out_layer.eval(feed_dict={self._model.rate: 0.0, self._pruning_factor: pruning_factor,
                                     self._learning_rate: 0.0,
                                     self._model.X: data['X_valid'], self._model.Rx: data['Rx_valid'],
                                     self._model.Rx_Y: data['Rx_Y_valid'], self._model.Tx_Y: data['Tx_Y_valid'],
                                     self._model.Tx: data['Tx_valid']})
                      Tx_valid = self._model.Tx.eval(feed_dict={self._model.rate: 0.0, self._pruning_factor: pruning_factor,
                                     self._learning_rate: 0.0,
                                     self._model.X: data['X_valid'], self._model.Rx: data['Rx_valid'],
                                     self._model.Rx_Y: data['Rx_Y_valid'], self._model.Tx_Y: data['Tx_Y_valid'],
                                     self._model.Tx: data['Tx_valid']})
                      _, Q = evaluate_output(Rx_valid, Tx_valid, M=16)
                      Q_set = np.append(Q_set, Q)

                  total_epoch = total_epoch + epoch_prune
              print("Testing Loss",
                    loss.eval(feed_dict={self._model.rate: 0.0, self._pruning_factor:pruning_factors[0], self._learning_rate: 0.0, self._model.X: data['X_test'],
                                         self._model.Rx: data['Rx_test'], self._model.Rx_Y: data['Rx_Y_test'],
                                         self._model.Tx_Y: data['Tx_Y_test'], self._model.Tx: data['Tx_test']}))

              Rx_test = self._model.out_layer.eval(
                  feed_dict={self._model.rate: 0.0, self._pruning_factor: pruning_factors[0],
                             self._learning_rate: 0.0,
                             self._model.X: data['X_test'], self._model.Rx: data['Rx_test'],
                             self._model.Rx_Y: data['Rx_Y_test'], self._model.Tx_Y: data['Tx_Y_test'],
                             self._model.Tx: data['Tx_test']})
              Tx_test = self._model.Tx.eval(
                  feed_dict={self._model.rate: 0.0, self._pruning_factor: pruning_factors[0],
                             self._learning_rate: 0.0,
                             self._model.X: data['X_test'], self._model.Rx: data['Rx_test'],
                             self._model.Rx_Y: data['Rx_Y_test'], self._model.Tx_Y: data['Tx_Y_test'],
                             self._model.Tx: data['Tx_test']})
              _, Q = evaluate_output(Rx_test, Tx_test, M=16)

              state = self._model.test_point.eval(
                  feed_dict={self._model.rate: 0.0, self._pruning_factor: pruning_factors[0],
                             self._learning_rate: 0.0,
                             self._model.X: data['X_test'], self._model.Rx: data['Rx_test'],
                             self._model.Rx_Y: data['Rx_Y_test'], self._model.Tx_Y: data['Tx_Y_test'],
                             self._model.Tx: data['Tx_test']})
              long_state = state[0, :, :]
              print("long state ", long_state.shape)

              np.savetxt('./long_state.csv', long_state, delimiter=',')
              print("Finished generating the long sequence.")

              np.savetxt('./training_loss_set.csv',training_loss_set, delimiter=',')
              np.savetxt('./testing_loss_set.csv',testing_loss_set, delimiter=',')
              np.savetxt('./Q_set.csv', Q_set,delimiter=',')

              learning_rate = 0.001
              NLCpred_train = self._model.get_output().eval(
                  feed_dict={self._model.rate: 0.0, self._learning_rate: learning_rate, self._pruning_factor:pruning_factors[0], self._model.X: data['X_test'],
                             self._model.Rx: data['Rx_test'], self._model.Rx_Y: data['Rx_Y_test'],
                             self._model.Tx_Y: data['Tx_Y_test'], self._model.Tx: data['Tx_test']})
              real = NLCpred_train[:, 0]
              image = NLCpred_train[:, 1]
              np.savetxt('./data_test_real.csv', real,delimiter=',')
              np.savetxt('./data_test_image.csv', image,delimiter=',')





