import torch
from torch import nn
from dataset import DatasetDNN
from model import Model
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import DataLoader
from scipy.io import loadmat


data_train = './Dataset_16QAM_1000_5WDM32G_PreCDC_train_2.mat'
data_test = './Dataset_16QAM_1000_5WDM32G_PreCDC_test_2.mat'
trip_train = './ADD_Triplets_16QAM_1000_PreCDC_5WDM32G_75_train_2.mat'
trip_test = './ADD_Triplets_16QAM_1000_PreCDC_5WDM32G_75_test_2.mat'
mul_trip_train = './Mul_Triplets_16QAM_1000_PreCDC_5WDM32G_75_train_2.mat‘
mul_trip_test = './Mul_Triplets_16QAM_1000_PreCDC_5WDM32G_75_test_2.mat'

learning_rate = 0.001
N_data=32768
pruning_iterations = 10
total_iterations = 3000
pruning_schedule = (np.ceil(np.ceil(2.0 ** (-np.arange(pruning_iterations, 0, -1)) * total_iterations)))
epochs = np.zeros(11)
pruning_factors = np.array([[0.1, 0.0], [0.15, 0.0], [0.2, 0.0], [0.25, 0.0], [0.3, 0.0], [0.35, 0.0], [0.4, 0.0], [0.45, 0.0], [0.5, 0.0], [0.55, 0.0], [0.6, 0.0]])
for idx, prune_step in enumerate(pruning_schedule):
    if idx==0:
        epochs[idx] = prune_step
    else:
        epochs[idx] = prune_step - pruning_schedule[idx-1]
epochs[10] = total_iterations-pruning_schedule[6]
epochs = epochs.astype('int32')
print("epochs", epochs)

def complex_rotation(x,y):
    sinx = np.sin(x[:,0])
    cosx = np.cos(x[:,0])
    yr = y[:, 0]
    yi = y[:, 1]
    return np.stack((cosx * yr - sinx * yi, cosx * yi + sinx * yr), axis=1)

def get_masks_for_model(model, pruning_factor):
    masks = []
    weight_matrices = []
    # Initialize Masks to Ones
    for name, params in model.named_parameters():
        if "0.weight" in name:
            masks.append(np.ones(params.shape))
            weight_matrices.append(params.detach().numpy())

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
        threshold = np.percentile(weights.ravel(), pruning_factor[idx] * 100)
        masks[idx][weights < threshold] = 0

### Print Pruning ratio
    final_pruning_factors = np.ones(len(masks))
    total_weights = 0
    pruned_weights = 0
    target_pruned_weights = 0

    for idx, mask in enumerate(masks):
        final_pruning_factors[idx] = np.count_nonzero(mask == 0) / mask.size
        total_weights += mask.size
        pruned_weights += np.count_nonzero(mask == 0)
        target_pruned_weights += (mask.size) * pruning_factor[idx]

    print("\nTotal Pruning Factor (Target): " + str(target_pruned_weights / total_weights))
    print("Total Pruning Factor (Final) : " + str(pruned_weights / total_weights))
    print("Target Pruning Factors For Each Layer:" + str(pruning_factor))
    print("Final Pruning Factors For Each Layer:" + str(final_pruning_factors) + "\n")

    return masks

def complex_mse_loss(output, target, batch_size):
    return (torch.abs(output-target)**2).sum()/batch_size

def train_loop(dataloader, model, masks, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)
    for batch, (X, X_Mul, Rx, Tx) in enumerate(dataloader):

        optimizer.zero_grad()

        # Apply Pruning Mask
        idx = 0
        with torch.no_grad():
            for name, params in model.named_parameters():
                if "0.weight" in name:
                    params[torch.from_numpy(masks[idx]) == 0] = 0
                    idx = idx + 1

        model.train()
        pred = model(X, X_Mul, Rx)
        loss = loss_fn(pred, Tx, batch_size=batch_size)

        # Backpropagation
        loss.backward()

        # Apply Pruning Mask To Corressponding Gradient
        idx = 0
        for name, params in model.named_parameters():
            if "0.weight" in name:
                params.grad[torch.from_numpy(masks[idx]) == 0] = 0
                idx = idx + 1
        optimizer.step()


def test(X_test, X_Mul_test, Rx_test, Tx_test, model, loss_fn):
    # evaluate model:
    model.eval()
    with torch.no_grad():
        pred = model(X_test, X_Mul_test, Rx_test)
        test_loss = loss_fn(pred, Tx_test, batch_size=len(Tx_test))

    print("Test loss is: ", test_loss)

class Trainer(object):
    def __init__(self, n_epochs, tr_batch_size, optimizer_params):
        self._n_epochs = n_epochs
        self._batch_size = tr_batch_size
        # Load the Dataset
        A = loadmat(trip_train)['Trip_real_X']
        A = np.array(A)
        Trip_real = A
        print("Trip_real", Trip_real.shape)

        A = loadmat(mul_trip_train)['Multi_Trip_real_X']
        A = np.array(A)
        Multi_Trip_real = A
        print("Multi_Trip_real_X", Multi_Trip_real.shape)

        A = loadmat(trip_train)['Trip_image_X']
        A = np.array(A)
        Trip_image = A


        A = loadmat(data_train)['Input_Symbol_real_X']
        A = np.array(A)
        Input_Symbol_Real = np.reshape(A, [-1, 1])
        A = loadmat(data_train)['Input_Symbol_image_X']
        A = np.array(A)
        Input_Symbol_Image = np.reshape(A, [-1, 1])
        A = loadmat(data_train)['Transmit_real_X']
        A = np.array(A)

        Transmit_Symbol_Real = np.reshape(A, [-1, 1])
        A = loadmat(data_train)['Transmit_image_X']
        A = np.array(A)
        Transmit_Symbol_Image = np.reshape(A, [-1, 1])
        self.X_train = np.c_[Trip_real, Trip_image]
        self.X_train = torch.from_numpy(self.X_train).float()
        self.X_Mul_train = Multi_Trip_real
        self.X_Mul_train = torch.from_numpy(self.X_Mul_train).float()

        self.Input_Symbol_Complex = np.c_[Input_Symbol_Real, Input_Symbol_Image]
        self.Input_Symbol_Complex = torch.from_numpy(self.Input_Symbol_Complex).float()
        self.Transmit_Symbol_Complex = np.c_[Transmit_Symbol_Real, Transmit_Symbol_Image]
        self.Transmit_Symbol_Complex = torch.from_numpy(self.Transmit_Symbol_Complex).float()


        # Testing
        A = loadmat(trip_test)['Trip_real_X']
        A = np.array(A)
        Trip_real_test = A

        A = loadmat(mul_trip_test)['Multi_Trip_real_X']
        A = np.array(A)
        Multi_Trip_real_test = A

        A = loadmat(trip_test)['Trip_image_X']
        A = np.array(A)
        Trip_image_test = A


        A = loadmat(data_test)['Input_Symbol_real_X']
        A = np.array(A)
        Input_Symbol_Real_test = np.reshape(A, [-1, 1])

        A = loadmat(data_test)['Input_Symbol_image_X']
        A = np.array(A)
        Input_Symbol_Image_test = np.reshape(A, [-1, 1])

        A = loadmat(data_test)['Transmit_real_X']
        A = np.array(A)
        Transmit_Symbol_Real_test = np.reshape(A, [-1, 1])

        A = loadmat(data_test)['Transmit_image_X']
        A = np.array(A)
        Transmit_Symbol_Image_test = np.reshape(A, [-1, 1])

        self.X_test = np.c_[Trip_real_test, Trip_image_test]
        self.X_test = torch.from_numpy(self.X_test).float()
        self.X_Mul_test = Multi_Trip_real_test
        self.X_Mul_test = torch.from_numpy(self.X_Mul_test).float()

        self.Input_Symbol_Complex_test_np = np.c_[Input_Symbol_Real_test, Input_Symbol_Image_test]
        self.Input_Symbol_Complex_test = torch.from_numpy(self.Input_Symbol_Complex_test_np).float()
        self.Transmit_Symbol_Complex_test = np.c_[Transmit_Symbol_Real_test, Transmit_Symbol_Image_test]
        self.Transmit_Symbol_Complex_test = torch.from_numpy(self.Transmit_Symbol_Complex_test).float()

        self._dataset_train = DatasetDNN(X=self.X_train, X_Mul=self.X_Mul_train, Rx=self.Input_Symbol_Complex, Tx=self.Transmit_Symbol_Complex)

        # self._model = torch.load("./model.ckpt")
        # print("Model Restored")
        self._model = Model()

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)

    def train_model(self):
        # loss function
        loss_fn = complex_mse_loss
        train_dataloader = DataLoader(self._dataset_train, batch_size=self._batch_size, shuffle=True)
        total_epoch = 0
       # Pruning steps
        for pruning_iteration in range(pruning_iterations+1):
            # How much to prune in this iteration
            pruning_factor = pruning_factors[pruning_iteration]
            print("Start Training For Pruning Factor: " + str(pruning_factor))

            masks = get_masks_for_model(self._model, pruning_factor)

            epoch = epochs[pruning_iteration]

            for t in range(epoch):
                print("Epoch is: ", total_epoch+t+1)
                train_loop(train_dataloader, self._model, masks, loss_fn, self._optimizer, self._batch_size)
                test(self.X_test, self.X_Mul_test, self.Input_Symbol_Complex_test, self.Transmit_Symbol_Complex_test, self._model, loss_fn)
            torch.save(self._model, "./modelPruned"+str(np.around(pruning_factor[0],2)*100)+"_"+str(np.around(pruning_factor[1],2)*100)+".ckpt")
            print("Model Saved")
            total_epoch = total_epoch + epoch
        print("Done!")
        test(self.X_test, self.X_Mul_test, self.Input_Symbol_Complex_test, self.Transmit_Symbol_Complex_test,
             self._model, loss_fn)
        with torch.no_grad():
            out_test = self._model(self.X_test, self.X_Mul_test, self.Input_Symbol_Complex_test).numpy()
            AD_NL = self._model.get_AD().numpy()
            Mul_NL = self._model.get_Mul().numpy()

        real = out_test[:, 0]
        image = out_test[:, 1]
        np.savetxt('./data_real.csv', real, delimiter=',')
        np.savetxt('./data_image.csv', image, delimiter=',')

        real = AD_NL[:, 0]
        image = AD_NL[:, 1]
        np.savetxt('./AD_NL_real.csv', real, delimiter=',')
        np.savetxt('./AD_NL_image.csv', image, delimiter=',')

        real = Mul_NL[:, 0]
        image = Mul_NL[:, 1]
        np.savetxt('./Mul_NL_real.csv', real, delimiter=',')
        np.savetxt('./Mul_NL_image.csv', image, delimiter=',')

        torch.save(self._model, "./PruneAll.ckpt")
        print("Model Saved")




