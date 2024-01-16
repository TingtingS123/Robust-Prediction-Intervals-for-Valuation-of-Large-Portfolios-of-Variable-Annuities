#!/usr/bin/env python
from mpi4py import MPI
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split
import time
import sys


def Huber(comm,rank, size):
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
    print(rank)
    y_test_pre_list = []
    error_model_list = []
    for i in range(5):
        Train_sample = Train.sample(frac = 1, replace = True)
        y_train_sample = Train_sample['fmv']
        X_train_sample = Train_sample.drop(['fmv'],axis = 1)

        model = HuberRegressor()
        model.fit(X_train_sample,y_train_sample)

        y_train_sample_pre = model.predict(X_train_sample)
        error_model = y_train_sample-y_train_sample_pre
        error_model_list.append(error_model)
        y_test_pre = model.predict(X_test)
        y_test_pre_list.append(y_test_pre)
        print('The rank',rank,'iteration',i)
    error_model_list = np.array(error_model_list)
    y_test_pre_list = np.array(y_test_pre_list)
    y_test_pre_l = np.empty((size,) + y_test_pre_list.shape)
    error_l = np.empty((size,) + error_model_list.shape)
    comm.Gatherv(error_model_list, error_l,root=0)
    comm.Gatherv(y_test_pre_list,y_test_pre_l, root=0)
    if rank == 0:
        y_test_pre_list_ = np.transpose(y_test_pre_l.reshape(y_test_pre_l.shape[0]*y_test_pre_l.shape[1],y_test_pre_l.shape[2]))
        # print(y_test_pre_list_.shape)
        error_model_list_ = np.transpose(error_l.reshape(error_l.shape[0]*error_l.shape[1],error_l.shape[2]))
        error_random = np.random.choice(np.array(error_model_list_).flatten(),(y_test_pre_list_.shape))
        y_test_revise_list   = y_test_pre_list_ +   error_random
        Left_list = np.quantile(np.transpose(y_test_revise_list),0.025,axis = 0)
        print(Left_list.shape)
        Right_list = np.quantile(np.transpose(y_test_revise_list),0.975,axis = 0)
        print(Right_list.shape)
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
    Huber(comm,rank, size)
    print((time.time()-t))
