# Deep Tensor Neural Network (DTNN)

This repository implements and reproduces results for the NN architecture DTNN. It uses QM9 dataset and train using 80% for train, 10% for validation and 10% for test.
Reference paper: https://arxiv.org/abs/1609.08259

## Instructions
- First clone the repository on a directory.
- Install required dependencies using requirements.txt file.
- Run data.py. It will first download the data and then filter it to create a mini dataset containing only molecules with less than or equal to 10 atoms.
- Run train.py to train the model on DTNN architecture on the mini dataset.
- Run test.py to test the model on test set.
