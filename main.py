#!/usr/bin/env python

import numpy as np  # Import NumPy for numerical operations.
import pandas as pd  # Import pandas for data manipulation and CSV file I/O.
import random  # Import random for random number generation.
from sklearn.model_selection import train_test_split  # Import train_test_split for splitting data into training and testing sets.
from sklearn import preprocessing  # Import preprocessing for data preprocessing utilities.
import matplotlib.pyplot as plt  # Import matplotlib.pyplot for plotting graphs.
import scipy.stats as stats  # Import scipy.stats for statistical functions.
from scipy.stats import tukey_hsd  # Import tukey_hsd for performing Tukey's Honest Significant Difference test.


def readfile(modelname, splitrate, resample):
    # Constructs the file path based on the model name, split rate, and resample iteration.
    path = "./" + modelname + "/result/"
    # Returns a DataFrame by reading the CSV file specified by the constructed path.
    return pd.read_csv(path + "results_" + str(splitrate) + "_" + str(resample) + ".csv")

def readata(rate, iteration, default=False):
    # Reads data for each specified model at a given rate and iteration.
    df_Huber = readfile("Huber", rate, iteration)
    df_RANSAC = readfile("RANSAC", rate, iteration)
    df_OLS = readfile("OLS", rate, iteration)
    df_quantile = readfile("quantile", rate, iteration)
    df_GradientBoost = readfile("GradientBoost", rate, iteration)
    # Returns DataFrames for each model.
    return df_OLS, df_Huber, df_RANSAC, df_quantile, df_GradientBoost
    
def getdensity(model, kde_x):
    # Calculates and returns the density estimate for a given model's data over a range of values.
    density = stats.gaussian_kde(model)
    return density(kde_x)



if __name__ == "__main__":
    rate_list = [10, 20, 30]  # Define different rates to iterate over.
    itera = [1000, 2000, 3000, 5000]  # Define different iteration counts.
    letter = ["(A)", "(B)", "(C)", "(D)"]  # Labels for subplots.
    
    # Iterate over each rate to generate plots.
    for rate in rate_list:
        h = 0  # Index for accessing the iteration counts.
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))  # Initialize a 2x2 grid of subplots.
    
        # Nested loops to fill each subplot with boxplot data.
        for i in range(2):
            for j in range(2):
                # Read data for the current rate and iteration.
                df_OLS, df_Huber, df_RANSAC, df_quantile, df_GradientBoost = readata(rate, itera[h])
                
                # Configure tick parameters for readability.
                ax[i, j].tick_params(axis='both', which='major', labelsize=7)
                ax[i, j].tick_params(axis='both', which='minor', labelsize=7)
                
                # Create a boxplot comparing the widths of prediction intervals across models.
                ax[i, j].boxplot([df_OLS['width'], df_Huber['width'], df_RANSAC['width'], df_quantile['width'], df_GradientBoost['width']])
                ax[i, j].set_xticklabels(["OLS", "Huber", "RANSAC", "quantile", "Gradient Boost"])
                ax[i, j].set_xlabel("{} R = {}".format(letter[h], itera[h]))
                
                # Perform Tukey's HSD test and print results.
                res = tukey_hsd(df_OLS['width'], df_Huber['width'], df_RANSAC['width'], df_quantile['width'], df_GradientBoost['width'])
                print("rate {}; iteration{}".format(rate, itera[h]))
                print(res)
                h += 1  # Increment the index to move to the next iteration count.
        
        # Set the overall title and save the figure.
        plt.title("Split Rate = {}%".format(rate))
        plt.savefig('./figure/rate_{}.png'.format(rate))

    side = 'Right_list'  # Define the variable to analyze in density plots.
    kde_x = np.linspace(-200, 1500, 1000)  # Define the x-axis range for density plots.
    
    # Iterate over each rate to generate density plots.
    for rate in rate_list:
        h = 0  # Index for accessing the iteration counts.
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))  # Initialize a 2x2 grid of subplots.
    
        # Nested loops to fill each subplot with density plot data.
        for i in range(2):
            for j in range(2):
                # Read data for the current rate and iteration.
                df_OLS, df_Huber, df_RANSAC, df_quantile, df_GradientBoost = readata(rate, itera[h])
                
                # Configure tick parameters for readability.
                ax[i, j].tick_params(axis='both', which='major', labelsize=7)
                ax[i, j].tick_params(axis='both', which='minor', labelsize=7)
                
                # Plot density estimates for the Right_list values of each model.
                ax[i, j].plot(kde_x, getdensity(df_OLS[side], kde_x), label="OLS")
                ax[i, j].plot(kde_x, getdensity(df_Huber[side], kde_x), label="Huber")
                ax[i, j].plot(kde_x, getdensity(df_RANSAC[side], kde_x), label="RANSAC")
                ax[i, j].plot(kde_x, getdensity(df_quantile[side], kde_x), label="quantile")
                ax[i, j].plot(kde_x, getdensity(df_GradientBoost[side], kde_x), label="Gradient Boost")
                ax[i, j].set_xlabel("{} R = {}".format(letter[h], itera[h]))
                
                # Perform Tukey's HSD test and print results
