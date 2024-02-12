#!/usr/bin/env python
from mpi4py import MPI  # Import MPI for parallel processing capabilities
import numpy as np  # Import NumPy for numerical operations
import pandas as pd  # Import pandas for data manipulation
# from sklearn.linear_model import HuberRegressor  # (Commented out, unused import)
from sklearn.linear_model import RANSACRegressor  # Import RANSACRegressor for robust linear modeling
from sklearn.model_selection import train_test_split  # (Unused import, could be removed)
import time  # Import time for execution timing
import sys  # Import sys for system-specific parameters and functions

def RANSACR(comm, rank, size):
    if rank == 0:
        data = pd.read_csv("data.csv")  # Root process reads the dataset
        split_rate = 0.9  # Define train-test split ratio
        # Split data into training and testing sets based on split_rate
        Train = data.iloc[:int(data.shape[0]*split_rate), :]
        Test = data.iloc[int(data.shape[0]*split_rate):, :]
    else:
        # Initialize Train and Test as None for non-root processes
        Train = None
        Test = None

    # Broadcast Train and Test datasets from the root process to all processes
    Train = comm.bcast(Train, root=0)
    Test = comm.bcast(Test, root=0)

    # Extract test labels and features
    y_test = Test['fmv']
    X_test = Test.drop(['fmv'], axis=1)
    print(rank)  # Print the rank of the current process

    y_test_pre_list = []  # List to store predictions for test set
    error_model_list = []  # List to store model errors

    for i in range(2):  # Iterate twice for sampling
        # Sample the training data with replacement
        Train_sample = Train.sample(frac=1, replace=True)
        y_train_sample = Train_sample['fmv']
        X_train_sample = Train_sample.drop(['fmv'], axis=1)

        # Initialize and fit the RANSAC regressor
        model = RANSACRegressor()
        model.fit(X_train_sample, y_train_sample)

        # Predict on training set to calculate error
        y_train_sample_pre = model.predict(X_train_sample)
        error_model = y_train_sample - y_train_sample_pre  # Calculate error
        error_model_list.append(error_model)  # Store error

        # Predict on test set
        y_test_pre = model.predict(X_test)
        y_test_pre_list.append(y_test_pre)  # Store predictions
        print('The rank', rank, 'iteration', i)  # Print current iteration and rank

    # Convert lists to NumPy arrays for aggregation
    error_model_list = np.array(error_model_list)
    y_test_pre_list = np.array(y_test_pre_list)
    
    # Prepare empty arrays for gathering results across processes
    y_test_pre_l = np.empty((size,) + y_test_pre_list.shape)
    error_l = np.empty((size,) + error_model_list.shape)

    # Gather errors and predictions from all processes to the root process
    comm.Gatherv(error_model_list, error_l, root=0)
    comm.Gatherv(y_test_pre_list, y_test_pre_l, root=0)

    if rank == 0:
        # Process the gathered data in the root process
        # Reshape and transpose for analysis
        y_test_pre_list_ = np.transpose(y_test_pre_l.reshape(y_test_pre_l.shape[0]*y_test_pre_l.shape[1], y_test_pre_l.shape[2]))
        print(y_test_pre_list_.shape)  # Print the shape of the processed predictions
        error_model_list_ = np.transpose(error_l.reshape(error_l.shape[0]*error_l.shape[1], error_l.shape[2]))

        # Randomly select errors to apply to predictions
        error_random = np.random.choice(np.array(error_model_list_).flatten(), (y_test_pre_list_.shape))
        # Adjust predictions by random error selection
        y_test_revise_list = y_test_pre_list_ + error_random

        # Calculate quantiles for prediction adjustment
        Left_list = np.quantile(np.transpose(y_test_revise_list), 0.025, axis=0)
        Right_list = np.quantile(np.transpose(y_test_revise_list), 0.975, axis=0)

        # Determine if actual values fall within predicted intervals
        Bool_list = (y_test >= Left_list) & (y_test < Right_list)

        # Calculate the width of prediction intervals
        diff = Right_list - Left_list

        # Prepare DataFrame to store results and save to CSV
        results_10_1000 = pd.DataFrame({
            'Left_list': np.array(Left_list),
            'Right_list': np.array(Right_list),
            'Bool_list': np.array(Bool_list),
            'width': np.array(diff),
            'test': np.array(Test['fmv'].copy())
        })
        pd.DataFrame(results_10_1000).to_csv('./result/results_10_1000.csv', index=False)
        print(f"accuracy: {Bool_list.mean():.6f}")  # Print the accuracy of interval predictions

if __name__ == "__main__":
    # Initialize MPI communication
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Get current process rank
    size = comm.Get_size()  # Get total number of processes
    t = time.time()  # Start timing
    RANSACR(comm, rank, size)  # Execute the main function
    print(time.time()-t)  # Print execution time
