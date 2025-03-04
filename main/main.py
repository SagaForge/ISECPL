'''
-------
main.py
-------

Module for implementing the whole pipeline of data cleaning and performance tuning

    * Random Search Baseline, given the path to the dataset, some budget b and minimising / maximising parameter. 

Returns a pandas data frame of the performance tuning metrics
    
Used by: none

'''

# LOCAL IMPORTS #
from data import DataCleaner ## data.py, cleans data prior to CPT
from baseline import RandomSearchBaseline ## baseline.py, for Random Search baseline
from tsampler import Sampler
from tuner import ConfigTuner, PerformanceLinkedList, FeatureNode

import os
import pandas as pd
import time  # For tracking run times
from tqdm import tqdm


issues = []


def baseline(path, runs, budgets):
    """
    Performs random search baseline for some provided budgets. For each budget provided, returns the best performance found for that budget
    """

    results = []

    for budget in budgets: # for every allocated budget

        budget_specific_performances = []

        for i in range (0, runs, 1):

            baseliner = RandomSearchBaseline(path, int(budget), performance_col=None, minimize=True)
            result = baseliner.random_search()

            budget_specific_performances.append(result[1]) ## apend performance of best config found to array

        results.append(min(budget_specific_performances)) # get best performance for the random search, and append to results

    
    return results


        
def hras(path, runs, budgets, initial_sample, mode):
    """
    Performs HRAS tuning for some provided budgets, initial sample, and mode. 
    For each budget provided, returns the best performance found for that budget.
    """
    results = []

    # Wrap the budgets loop with tqdm for a progress bar
    for budget in tqdm(budgets, desc="Processing Budgets"):  # for every allocated budget
        budget_specific_performances = []

        # Wrap the runs loop with tqdm for a nested progress bar
        for i in tqdm(range(runs), desc=f"Runs for Budget {budget}", leave=False):
            hras = ConfigTuner(dataset_path=path, budget=budget, initial_sample=initial_sample, performance_col=None)
            hras.run(mode=mode)

            budget_specific_performances.append(hras.best_performance)  # append performance of best config found to array

        results.append(min(budget_specific_performances))  # get best performance for the random search, and append to results

    print(f"[main.py] <INFO> HRAS Tuning Complete")

    return results



def main():
    # Specify the path to the dataset
    datasets_path = '/home/connor/university/isecpl/datasets/' # path to datasets folder

    # Create an empty DataFrame with the desired columns
    results_df = pd.DataFrame(columns=[
        'Dataset', 'Budget', 
        'Baseline', 'HRAS_Standard', 'HRAS_A', 'HRAS_AE', 'HRAS_AG', # Performance columns
        'Baseline_Time', 'HRAS_Standard_Time', 'HRAS_A_Time', 'HRAS_AE_Time', 'HRAS_AG_Time',  # Time columns
        'Best_Algorithm'  # Column to indicate the best algorithm
    ])

    runs = 1    # how many runs to do per budget 
    budgets = [500] # what budgets to iterate through

    hras_standard_is = 0.4
    hras_adaptive_is = 0.4
    hras_ae_is = 0.4
    hras_adaptive_gaussian_is = 0.4

    for filename in os.listdir(datasets_path):
        file_path = os.path.join(datasets_path, filename)

        if os.path.isfile(file_path): # existence verified, continue with cleaning data

            try:
                cleaner = DataCleaner(file_path)
                cleaned_data = cleaner.clean_data()

                # Data Now Cleaned, continue with running CPT for each strategy

                # Track run times and results for each algorithm
                start_time = time.time()
                baseline_results = baseline(cleaned_data, runs, budgets) # gets best performances for each budget for random search
                baseline_time = time.time() - start_time

                start_time = time.time()
                hras_standard_results = hras(cleaned_data, runs, budgets, hras_standard_is, 1) # gets best performance for each budget for hras (non explorative and non adaptive)
                hras_standard_time = time.time() - start_time

                start_time = time.time()
                hras_adaptive_results = hras(cleaned_data, runs, budgets, hras_adaptive_is, 2) # gets best performance for each budget for hras (adaptive)
                hras_adaptive_time = time.time() - start_time

                start_time = time.time()
                hras_adaptive_explorative_results = hras(cleaned_data, runs, budgets, hras_ae_is, 3) # gets best performance for each budget for hras (adaptive and explorative)
                hras_adaptive_explorative_time = time.time() - start_time

                start_time = time.time()
                hras_adaptive_gaussian_results = hras(cleaned_data, runs, budgets, hras_adaptive_gaussian_is, 4)
                hras_adaptive_gaussian_time = time.time() - start_time

                # Iterate through budgets and add rows to the DataFrame
                for i, budget in enumerate(budgets):
                    # Determine the best algorithm (smallest performance value)
                    performances = {
                        'Baseline': baseline_results[i],
                        'HRAS_Standard': hras_standard_results[i],
                        'HRAS_A': hras_adaptive_results[i],
                        'HRAS_AE': hras_adaptive_explorative_results[i],
                        'HRAS_AG': hras_adaptive_gaussian_results[i]
                    }
                    best_algorithm = min(performances, key=performances.get)

                    # Create a new row
                    new_row = pd.DataFrame({
                        'Dataset': [filename],  # Pass as a list
                        'Budget': [budget],     # Pass as a list
                        'Baseline': [baseline_results[i]],  # Pass as a list
                        'HRAS_Standard': [hras_standard_results[i]],  # Pass as a list
                        'HRAS_A': [hras_adaptive_results[i]],  # Pass as a list
                        'HRAS_AE': [hras_adaptive_explorative_results[i]],  # Pass as a list
                        'HRAS_AG': [hras_adaptive_gaussian_results[i]],
                        'Baseline_Time': [baseline_time],  # Pass as a list
                        'HRAS_Standard_Time': [hras_standard_time],  # Pass as a list
                        'HRAS_A_Time': [hras_adaptive_time],  # Pass as a list
                        'HRAS_AE_Time': [hras_adaptive_explorative_time],  # Pass as a list
                        'HRAS_AG_Time': [hras_adaptive_gaussian_time],
                        'Best_Algorithm': [best_algorithm]  # Pass as a list
                    })

                    # Append the new row to the results DataFrame using pd.concat
                    results_df = pd.concat([results_df, new_row], ignore_index=True)

                print(f"Processed dataset: {filename}")

                # Cleanup and prepare for next dataset to tune
                os.remove(cleaned_data)

            except Exception as e:
                issues.append(e)
                
                continue ## 

    return results_df

if __name__ == "__main__":
    results = main()
    print("<< RESULTS >>")
    print(results)
    print("With the following issues: ")
    for i in issues:
        print(issues)