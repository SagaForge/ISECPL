# Hierarchal Gaussian Adaptive Sampling with Diversification for Configuration Performance Tuning #
## ISE Tool Building Project Submission ##
## SID: 2120773 ##


## Introduction ##

This project implements a Configuration Performance Tuner for the Intelligent Software Engineering module, UoB. Usage and instructions (the required *manual*, *replication* and *requirements* submissions) can be found under the *pdfs* folder. 

If there are issues with the code, please check Python dependencies and file paths (declared in main.py and data.py).

This project includes the datasets to be automatically tested. The CPT automatically iterates through any csv files found in the *datasets* folder, so to tune specific datasets, remove or hide the csv files appropriatley.


## *main* modules ## 

•	data.py – Responsible for cleaning configuration datasets prior to performance tuning. It leverages standard regex text cleaning and character removal, outlier detection and removal using Z-Scores + IQR and imputation of missing values using column mean average (CMA) for standardised CPT.

•	tsampler.py – Module responsible for blind and observed sampling from configuration space, dynamic diversified initial sample allocations and budget management.

•	baseline.py – Simple implementation of the Random Search baseline, for comparisons to HGAS.

•	tuner.py – The module responsible for implementing the HGAS strategy itself. Implements the hierarchal logical structures, sampling strategies and misc. strategies for comparisons (explained below).

•	main.py – Parent module for implementing and executing the entire cleaning →         sampling → tuning → results pipeline.





**The two contributors are myself from separate accounts (accidental), all work is authentic, original and individual.**
