#!/usr/bin/env python
from mpi4py import MPI  # Import MPI for parallel processing capabilities.
import numpy as np  # Import NumPy for numerical operations.
import pandas as pd  # Import pandas for data manipulation and CSV file I/O.
from sklearn.ensemble import GradientBoostingRegressor  # Import GradientBoostingRegressor for regression tasks.
from sklearn.model_selection import train_test_split  # (Unused import, could be removed)
import time  # Import time to measure execution duration.
import sys  # Import sys for interacting with the Python interpreter.

def GBoost(comm, rank, size):
    # Define the Gradient Boosting function to distribute tasks across processors.
    if rank == 0:
        # The root process reads the dataset and performs the train-test split.
        data = pd.read_csv("data.csv")
        split_rate = 0.9  # Define the proportion of data for training.
        Train = data.iloc[:int(data.shape[0]*split_rate), :]
        Test = data.iloc[int(data.shape[0]*split_rate):, :]
    else:
        # Non-root processes initialize Train and Test as None.
        Train = None
        Test = None

    # Broadcast the Train and Test datasets from the root process to all processes.
    Train = comm.bcast(Train, root=0)
    Test = comm.bcast(Test, root=0)

    # Prepare the test dataset for model prediction.
    y_test = Test['fmv']
    X_test = Test.drop(['fmv'], axis=1)
    print(rank)  # Print the rank of the current process.

    y_test_pre_list = []  # List to store predictions for the test dataset.
    error_model_list = []  # List to store model prediction errors.
    iters = 100  # Define the number of bootstrap iterations.

    for i in range(iters):
        # Perform bootstrapping: Sample the training data with replacement.
        Train_sample = Train.sample(frac=1, replace=True)
        y_train_sample = Train_sample['fmv']
        X_train_sample = Train_sample.drop(['fmv'], axis=1)

        # Initialize and fit the Gradient Boosting regressor.
        model = GradientBoostingRegressor()
        model.fit(X_train_sample, y_train_sample)

        # Predict on the training set to calculate model prediction error.
        y_train_sample_pre = model.predict(X_train_sample)
        error_model = y_train_sample - y_train_sample_pre
        error_model_list.append(error_model)  # Store model errors.
        y_test_pre = model.predict(X_test)
        y_test_pre_list.append(y_test_pre)  # Store test predictions.
        print('The rank', rank, 'iteration', i)  # Print current iteration and rank.

    # Convert lists to NumPy arrays for collective operations.
    error_model_list = np.array(error_model_list)
    y_test_pre_list = np.array(y_test_pre_list)

    # Prepare empty arrays to gather results from all processes at the root.
    y_test_pre_l = np.empty((size,) + y_test_pre_list.shape)
    error_l = np.empty((size,) + error_model_list.shape)

    # Gather prediction errors and predictions at the root process.
    comm.Gatherv(error_model_list, error_l, root=0)
    comm.Gatherv(y_test_pre_list, y_test_pre_l, root=0)

    if rank == 0:
        # Process gathered data at the root process for analysis.
        y_test_pre_list_ = np.transpose(y_test_pre_l.reshape(y_test_pre_l.shape[0]*y_test_pre_l.shape[1], y_test_pre_l.shape[2]))
        print(y_test_pre_list_.shape)  # Print the shape of the predictions array.
        error_model_list_ = np.transpose(error_l.reshape(error_l.shape[0]*error_l.shape[1], error_l.shape[2]))

        # Apply randomly selected errors to the test predictions to simulate prediction variance.
        error_random = np.random.choice(np.array(error_model_list_).flatten(), (y_test_pre_list_.shape))
        y_test_revise_list = y_test_pre_list_ + error_random

        # Calculate prediction intervals from the adjusted predictions.
        Left_list = np.quantile(np.transpose(y_test_revise_list), 0.025, axis=0)
        Right_list = np.quantile(np.transpose(y_test_revise_list), 0.975, axis=0)
        Bool_list = (y_test >= Left_list) & (y_test < Right_list)  # Check if actual values fall within intervals.

        # Calculate the width of the prediction intervals.
        diff = Right_list - Left_list

        # Store results in a DataFrame and save to a CSV file for further analysis.
        results_10_1000 = pd.DataFrame({
            'Left_list': np.array(Left_list),
            'Right_list': np.array(Right_list),
            'Bool_list': np.array(Bool_list),
            'width': np.array(diff),
            'test': np.array(Test['fmv'].copy())
        })
        pd.DataFrame(results_10_1000).to_csv('./result/results_10_1000.csv', index=False)
        print(f"accuracy: {Bool_list.mean():.6f}")  # Print the accuracy of the prediction intervals.

if __name__ == "__main__":
    # Initialize MPI communication, get the rank and size of the process, and measure execution time.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    t = time.time()  # Start timing.
    GBoost(comm, rank, size)  # Execute the Gradient Boosting function across multiple processes.
    print(time.time()-t)  # Print the total execution time.
