To re-simulate the Bi_RNN_Feature_Generation_Network
1. Preprocessing the training dataset, validation dataset and testing dataset to create a sliding window
	Runing the Preprocessing_X_train.m and Preprocessing_Y_train.m to preprocess the training dataset
	Runing the Preprocessing_X_test.m, Preprocessing_Y_test.m to preprocess the validation and testing dataset (change the input and output file name for validation and testing dataset)

2. Train and prune the bi-RNN network
	Training the network by runing the Tensorflow file in Tensorflow_train_and_pruning_code: Run_script.py
	Initially, change the pruning_factors to all 0s and set the total_iterations to about 3000 to train it without pruning.
	After It converges, save the model and start to prune by setting pruning_factors in trainer.py to an increasing order and increasing the total_iterations to about 30000 or even higher
	Pruning factors should be increased more slowly as the pruning processed
	The total_iterations should be increased as the pruning processed and the learning rate could be slowly reduced as the pruning processed
	Saving the model as the training processed
	Sometimes, the loss and Q factor will dramatically drop, then stop the training and either restarting it or starting from a point where the drop hasn't come

3. Quick test the bi-RNN network
	(1) Appending 0s at beginning and end by running the MATLAB code: Append_zeros.m
	(2) Generate the hidden state for the testing sequence by running the Tensorflow code: Run_script.py in Tensorflow_hidden_state_generation
	(3) Process the hidden states by running the MATLAB code: Process_state.m
	(4) Generate the estimated nonlinear distortion for the testing sequence by running the Tensorflow code: Run_script.py in Tensorflow_final_result
	(5) Runing the Q_Evaluation.m in MATLAB_CODE to calculate the Q factor

4. Quantization
	(1) Saving the trained weights by running the Tensorflow code: Run_script.py in Tensorflow_final_result
	(2) Run the quantization by running the MATLAB code: Quantization
