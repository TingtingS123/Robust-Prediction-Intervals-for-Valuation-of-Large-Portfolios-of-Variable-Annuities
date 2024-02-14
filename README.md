# Robust Prediction Intervals for Valuation of Large Portfolios of Variable Annuities: A comparative Study of Five Models

In this article, we explored the generation of robust prediction intervals for variable annuity pricing using five distinct models enhanced by bootstrapping techniques. Our analysis revealed that the Gradient Boosting Regression model offers the most optimal balance between interval narrowness and coverage rate, making it the recommended approach for the accurate valuation of large variable annuity portfolios.

## Description
Valuation of large portfolios of variable annuities (VAs) is a well-researched area in actuarial science field. However, the study of producing reliable prediction intervals for prices has received comparatively less attention. Compared to point prediction, the prediction interval can calculate a reasonable price range of VAs and helps investors and insurance companies better manage risk in order to maintain profitability and sustainability. 

In this study, we address this gap by utilizing five different models in conjunction with bootstrapping techniques to generate robust prediction intervals for variable annuity prices. 
Our findings show that the Gradient Boosting regression (GBR) model provides the narrowest intervals compared to the other four models. While the  Random sample consensus (RANSAC) model has the highest coverage rate, but it has the widest interval. In practical applications, considering the trade-off between coverage rate and interval width, the GBR model would be a preferred choice.

Therefore, we recommend using the gradient boosting model with the bootstrap method to calculate the prediction interval of valuation for a large portfolio of variable annuity policies.

## Getting Started

### Data load
The details of the dataset used in this paper can be found at the following URL: https://www2.math.uconn.edu/~gan/software.html

### Libraries Used
Libraries used: Numpy, Pandas, mpi4py, sklearn, matplotlib, scipy, statsmodels.

How to install the libraries used:
```bash
pip install -r requirements.txt
```

### How to employ mpi4py for executing programs in parallel

Utilizing mpi4py for parallel processing of large datasets is crucial for enhancing computational efficiency and saving time. mpi4py is a Python library based on the Message Passing Interface (MPI) that enables programs to run in parallel across multiple processors. This parallel execution allows for the division of tasks across multiple computing nodes, accelerating the processing and analysis of data. 

The following SBATCH directives are used to configure the job's resource requirements and runtime environment:
```bash
#SBATCH --job-name=bootstrap assigns a unique name to the job for easier identification.
#SBATCH --nodes=4 specifies the job will use four computing nodes.
#SBATCH --ntasks=10 determines that the job will run ten tasks in parallel.
#SBATCH --cpus-per-task=1 allocates one CPU per task, optimizing the distribution of computational workload.
#SBATCH --time=12:00:00 limits the total runtime to 12 hours, ensuring the job completes within a reasonable timeframe.
#SBATCH --partition=research assigns the job to a 'research' partition, which is a subset of the cluster tailored for research tasks.
#SBATCH --output=%x.%j.out and #SBATCH --error=%x.%j.err direct the job's standard output and error messages to specific files, using the job name and job ID for file naming.
```
These directives are essential for efficiently scheduling the parallel computation job on an HPC cluster, enabling the rapid and effective analysis of large datasets with mpi4py.

```bash
sbatch run_mpi4.sh
```
###


### Environment Infomation
python version: 3.6.8

10 general purpose CPU Nodes, each with (2) Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz (Intel Broadwell), 64GB RAM, and 1 TB of scratch space. (Total 320 threads across 10 nodes)
 
2 CPU high memory nodes, each with (2) Intel(R) Xeon(R) Gold 6254 CPU @ 3.10GHz (Intel Cascade Lake), 768GB RAM, and 14TB of local storage (RAID 1) for scratch/temp storage. (Total 144 threads across 2 nodes)
 
4 GPU nodes with (2) Intel(R) Xeon(R) CPU E5-2687W @ 3.10GHz (Intel Sandy Bridge EP), 64GB RAM, 2x NVIDIA 2080Ti’s, and 1 TB of SSD scratch space. (Total 128 threads across 4 nodes).
 
5 GPU nodes with (1) Intel(R) Xeon(R) W-2255 CPU @ 3.70GHz (Intel Cascade Lake), 64GB RAM, 1x NVIDIA A5000, and 1TB of NVMe SSD scratch space (100 threads across 5 nodes).
 
Summarizing, babbage is a 21 node cluster located in the library server room. It consists of 692 CPU threads, 2752GB of RAM, 13 GPUs, and approximately 80TB of centralized storage.
