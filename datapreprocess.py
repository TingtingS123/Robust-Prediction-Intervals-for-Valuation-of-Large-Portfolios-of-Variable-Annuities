#!/usr/bin/env python

import numpy as np  # Import NumPy for numerical operations.
import pandas as pd  # Import pandas for data manipulation and reading CSV files.
import random  # Import random for random operations like seeding.

if __name__ == "__main__":
    # Data preprocessing steps.
    
    # Read the Greek and inforce datasets from CSV files.
    Greek = pd.read_csv(r'Greek.csv')
    inforce = pd.read_csv('inforce.csv')
    
    # Merge the two datasets on common columns.
    df = pd.merge(Greek, inforce)
    
    # Calculate 'age' and 'time to maturity (ttm)' in years.
    age = (df['currentDate'] - df['birthDate']) / 365
    ttm = (df['matDate'] - df['currentDate']) / 365
    
    # Select relevant columns for further analysis and add 'age' and 'ttm' to the dataframe.
    df = df[['recordID', 'gender', 'gmwbBalance', 'productType', 'gbAmt', 'FundValue1', 'FundValue2', 'FundValue3', 'FundValue4', 'FundValue5',
             'FundValue6', 'FundValue7', 'FundValue8', 'FundValue9', 'FundValue10', 'fmv']]
    df["age"] = age
    df['ttm'] = ttm
    df = df[['recordID', 'gender', 'gmwbBalance', 'productType', 'gbAmt', 'FundValue1', 'FundValue2', 'FundValue3', 'FundValue4', 'FundValue5',
             'FundValue6', 'FundValue7', 'FundValue8', 'FundValue9', 'FundValue10', 'fmv', 'age', 'ttm']]
    
    # Generate dummy variables for categorical variables and merge them with the original dataframe.
    dummy = pd.get_dummies(df[['recordID', 'productType', 'gender']], drop_first=True)
    df = pd.merge(df, dummy)
    
    # Drop original columns of categorical variables after encoding.
    df = df.drop(['productType', 'gender', 'recordID'], axis=1)
    
    # List of column names to be normalized.
    colname = ['gmwbBalance', 'gbAmt', 'FundValue1', 'FundValue2', 'FundValue3',
               'FundValue4', 'FundValue5', 'FundValue6', 'FundValue7', 'FundValue8',
               'FundValue9', 'FundValue10']
    
    # Normalize specified columns by their range and mean.
    df[colname] = (df[colname] - df[colname].mean(0)) / (df[colname].max(0) - df[colname].min(0))
    
    # Normalize 'fmv' column by dividing by 1000.
    df['fmv'] = df['fmv'] / 1000
    
    # Seed the random number generator for reproducibility and shuffle the dataframe.
    random.seed(10)
    df = df.sample(frac=1, replace=False)
    
    # Save the processed dataframe to a new CSV file.
    df.to_csv('data.csv', index=False)
