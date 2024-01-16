from mpi4py import MPI
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import sys
import statsmodels.regression.quantile_regression as Q_reg


def quantile(comm,rank, size):
    if rank == 0:
        data = pd.read_csv("data.csv")
        split_rate = 0.9
        Train = data.iloc[:int(data.shape[0]*split_rate),:]
        Test = data.iloc[int(data.shape[0]*split_rate):,:]

    else:
        # data = None
        Train = None
        Test = None
    # data = comm.bcast(data,root = 0)
    Train = comm.bcast(Train,root = 0)
    Test = comm.bcast(Test,root = 0)


    y_test = Test['fmv']
    X_test = Test.drop(['fmv'],axis = 1)
    # print(rank)
    y_test_pre_left = []
    y_test_pre_right = []
    for i in range(3):
        print('The rank',rank,'iteration',i)
        Train_sample = Train.sample(frac = 1, replace = True)
        y_train_sample = Train_sample['fmv']
        X_train_sample = Train_sample.drop(['fmv'],axis = 1)
        Y_test_pred_left = Q_reg.QuantReg(y_train_sample, X_train_sample).fit(q=0.025).predict(X_test)
        Y_test_pred_right = Q_reg.QuantReg(y_train_sample, X_train_sample).fit(q=0.975).predict(X_test)
        y_test_pre_left.append(Y_test_pred_left)
        y_test_pre_right.append(Y_test_pred_right)
        

    y_test_pre_left = np.array(y_test_pre_left)
    y_test_pre_l = np.empty((size,) +y_test_pre_left.shape)
    y_test_pre_right = np.array(y_test_pre_right)
    y_test_pre_r = np.empty((size,) +y_test_pre_right.shape)
    comm.Gatherv(y_test_pre_left, y_test_pre_l,root=0)
    comm.Gatherv(y_test_pre_right,y_test_pre_r, root=0)
    if rank == 0:
        y_test_pre_l_list = np.transpose(y_test_pre_l.reshape(y_test_pre_l.shape[0]*y_test_pre_l.shape[1],y_test_pre_l.shape[2]))
        y_test_pre_r_list = np.transpose(y_test_pre_r.reshape(y_test_pre_r.shape[0]*y_test_pre_r.shape[1],y_test_pre_r.shape[2]))
        Left_list = np.quantile(np.transpose(y_test_pre_l_list),0.5,axis = 0)
        Right_list = np.quantile(np.transpose(y_test_pre_r_list),0.5,axis = 0)
        Bool_list = (y_test >= Left_list) & (y_test< Right_list)
        diff = Right_list-Left_list
        results_10_1000 = pd.DataFrame([])
        results_10_1000['Left_list'] = np.array(Left_list)
        results_10_1000['Right_list'] = np.array(Right_list)
        results_10_1000['Bool_list'] = np.array(Bool_list)
        results_10_1000['width'] = np.array(diff)
        results_10_1000['test'] = np.array(Test['fmv'].copy())
        pd.DataFrame(results_10_1000).to_csv('./result/results_10_1000.csv',index = False)
        print(f"accuracy: {Bool_list.mean():.6f}")


if __name__ == "__main__":
    # name = MPI.Get_processor_name()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    t = time.time()
    quantile(comm,rank, size)
    print(time.time()-t)
