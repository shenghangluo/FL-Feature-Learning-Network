To run the NN-PB-NLC
1. Generate triplets for both training dataset and testing dataset.
	triplets for training dataset is generated by running the MATLAB file in MATLAB_CODE: FIRST_ORDER_TRIPLET_CALCULATION_train.m
	triplets for testing dataset is generated by running the MATLAB file in MATLAB_CODE: FIRST_ORDER_TRIPLET_CALCULATION_test.m
3. Train the NN network
	Training the network by runing the pytorch file in Pytorch_code: Run_script.py
	Initially training the network without pruning. Set pruning_factors in trainer.py to all 0s and reduce the total_iterations to about 300
	Save the trained model checkpoint by enabling torch.save(self._model) in trainer.py
4. Pruning the NN network
	Restore the trained model by enabling self._model = torch.load("./model.ckpt") and disabling self._model = Model()
	Set pruning_factors in trainer.py to an increasing order and increase the total_iterations to about 3000
	Pruning factors should be increased more slowly as the pruning processed
	The learning rate could be slowly reduced as the pruning processed
	Save the trained model checkpoint by enabling torch.save(self._model) in trainer.py
4. Q factor Evaluation
	Runing the Q_Evaluation.m in MATLAB_CODE to calculate the Q factor.