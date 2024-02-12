from mpi4py import MPI  # Import MPI for parallel computing capabilities.
import numpy as np  # Import NumPy for numerical operations.
import pandas as pd  # Import pandas for data manipulation.
from sklearn.model_selection import train_test_split  # (Unused import, could be removed)
import time  # Import time for timing executions.
import sys  # Import sys for system-specific parameters and functions.
import statsmodels.regression.quantile_regression as Q_reg  # Import Quantile Regression from statsmodels.

def quantile(comm, rank, size):
    if rank == 0:
        data = pd.read_csv("data.csv")  # The root process reads the data.
        split_rate = 0.9  # Define the train-test split ratio.
        # Split data into training and testing sets based on split_rate.
        Train = data.iloc[:int(data.shape[0]*split_rate),:]
        Test = data.iloc[int(data.shape[0]*split_rate):,:]
    else:
        # Initialize Train and Test as None for non-root processes.
        Train = None
        Test = None
    # Broadcast Train and Test datasets from the root process to all processes.
    Train = comm.bcast(Train, root=0)
    Test = comm.bcast(Test, root=0)

    # Extract test labels and features.
    y_test = Test['fmv']
    X_test = Test.drop(['fmv'], axis=1)
    
    # Initialize lists to store predictions.
    y_test_pre_left = []
    y_test_pre_right = []
    iters = 100  # Define the number of bootstrap iterations.

    for i in range(iters):
        print(f'The rank {rank}, iteration {i}')
        # Sample the training data with replacement.
        Train_sample = Train.sample(frac=1, replace=True)
        y_train_sample = Train_sample['fmv']
        X_train_sample = Train_sample.drop(['fmv'], axis=1)
        # Perform quantile regression and predict on the test set.
        Y_test_pred_left = Q_reg.QuantReg(y_train_sample, X_train_sample).fit(q=0.025).predict(X_test)
        Y_test_pred_right = Q_reg.QuantReg(y_train_sample, X_train_sample).fit(q=0.975).predict(X_test)
        # Append predictions to lists.
        y_test_pre_left.append(Y_test_pred_left)
        y_test_pre_right.append(Y_test_pred_right)

    # Convert predictions lists to arrays.
    y_test_pre_left = np.array(y_test_pre_left)
    y_test_pre_right = np.array(y_test_pre_right)
    # Initialize empty arrays to gather results across processes.
    y_test_pre_l = np.empty((size,) + y_test_pre_left.shape)
    y_test_pre_r = np.empty((size,) + y_test_pre_right.shape)
    # Gather predictions from all processes to the root process.
    comm.Gatherv(y_test_pre_left, y_test_pre_l, root=0)
    comm.Gatherv(y_test_pre_right, y_test_pre_r, root=0)
    
    if rank == 0:
        # Process and save the results in the root process.
        # Reshape and transpose prediction arrays for analysis.
        y_test_pre_l_list = np.transpose(y_test_pre_l.reshape(y_test_pre_l.shape[0]*y_test_pre_l.shape[1], y_test_pre_l.shape[2]))
        y_test_pre_r_list = np.transpose(y_test_pre_r.reshape(y_test_pre_r.shape[0]*y_test_pre_r.shape[1], y_test_pre_r.shape[2]))
        # Calculate median quantiles and determine intervals.
        Left_list = np.quantile(np.transpose(y_test_pre_l_list), 0.5, axis=0)
        Right_list = np.quantile(np.transpose(y_test_pre_r_list), 0.5, axis=0)
        # Check if actual values fall within predicted intervals.
        Bool_list = (y_test >= Left_list) & (y_test < Right_list)
        diff = Right_list - Left_list  # Calculate the width of intervals.
        # Create a DataFrame to store results and save it to a CSV file.
        results_10_1000 = pd.DataFrame({
            'Left_list': np.array(Left_list),
            'Right_list': np.array(Right_list),
            'Bool_list': np.array(Bool_list),
            'width': np.array(diff),
            'test': np.array(Test['fmv'].copy())
        })
        pd.DataFrame(results_10_1000).to_csv('./result/results_10_1000.csv', index=False)
        print(f"accuracy: {Bool_list.mean():.6f}")  # Print the accuracy of interval predictions.

