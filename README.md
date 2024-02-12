# Robust Prediction Intervals for Valuation of Large Portfolios of Variable Annuities: A comparative Study of Five Models

In this article, we explored the generation of robust prediction intervals for variable annuity pricing using five distinct models enhanced by bootstrapping techniques. Our analysis revealed that the Gradient Boosting Regression model offers the most optimal balance between interval narrowness and coverage rate, making it the recommended approach for the accurate valuation of large variable annuity portfolios.

Authors and contributors: Tingting Sun, Haoyuan Wang, Donglin Wang(Advisor).

## Description
Valuation of large portfolios of variable annuities (VAs) is a well-researched area in actuarial science field. However, the study of producing reliable prediction intervals for prices has received comparatively less attention. Compared to point prediction, the prediction interval can calculate a reasonable price range of VAs and helps investors and insurance companies better manage risk in order to maintain profitability and sustainability. 

In this study, we address this gap by utilizing five different models in conjunction with bootstrapping techniques to generate robust prediction intervals for variable annuity prices. 
Our findings show that the Gradient Boosting regression (GBR) model provides the narrowest intervals compared to the other four models. While the  Random sample consensus (RANSAC) model has the highest coverage rate, but it has the widest interval. In practical applications, considering the trade-off between coverage rate and interval width, the GBR model would be a preferred choice.

Therefore, we recommend using the gradient boosting model with the bootstrap method to calculate the prediction interval of valuation for a large portfolio of variable annuity policies.



## Getting Started
# 讲解一下为什么要用mpi4py

### Data load
The details of the dataset used in this paper can be found at the following URL: https://www2.math.uconn.edu/~gan/software.html

### Libraries used
Libraries used: Numpy, Pandas, mpi4py, sklearn, sys, time(not necessary, just for recording the processing time), matplotlib.pyplot, scipy.stats, statsmodels.api,sys, statsmodels.regression.quantile_regression. 

Example about how to install the library:
```bash
pip3 install Numpy

### How to use mpi4y



## 感谢部分
