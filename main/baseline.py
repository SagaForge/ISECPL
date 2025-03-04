'''
-------
baseline.py
-------

Module for implementing a simple Random Search baseline for CPT prior to tuning. Implements:

    * Random Search Baseline, given the path to the dataset, some budget b and minimising / maximising parameter. 

Returns the entire, best configurgation found by Random Search. 
    
Used by: main.py

'''

import pandas as pd
import numpy as np

class RandomSearchBaseline:
    def __init__(self, dataset_path, budget, performance_col=None, minimize=True):
        """
        Initializes the Random Search Baseline Tool.

        :param dataset_path: Path to the dataset (CSV format assumed)
        :param budget: Maximum number of measurements allowed
        :param performance_col: Optional, Name of the performance column, assumes last index if not specified
        :param minimize: Boolean indicating whether to minimize or maximize performance
        """
        self.dataset = pd.read_csv(dataset_path)
        self.budget = int(min(budget, len(self.dataset)))  # Ensure budget does not exceed dataset size
        self.minimize = minimize

        # Remove Interaction Columns (as this is the baseline, no need to analyse interaction features)
        self.dataset = self.dataset.loc[:, ~self.dataset.columns.str.contains('interaction')]

        if performance_col:
            self.performance_col = performance_col
        else:
            #print("[baseline.py] <INFO> No column provided, assuming last table column as performance metric")
            # Assume last column is performance and drop 'interaction' columns
            
            self.performance_col = self.dataset.columns[-1]
            #print("[baseline.py] <INFO> Column Metric Found: ", self.dataset.columns[-1])

        self.best_config = None
        self.best_performance = np.inf if minimize else -np.inf
        
        # Select feature columns, excluding performance column
        self.feature_columns = [col for col in self.dataset.columns if col != self.performance_col]

    def random_search(self):
        """
        Performs Random Search within the allocated budget.
        """
        sampled_indices = np.random.choice(self.dataset.index, self.budget, replace=False)

        for idx in sampled_indices:
            config = self.dataset.loc[idx, self.feature_columns].to_dict()
            performance = self.dataset.loc[idx, self.performance_col]

            if self._is_better(performance):
                self.best_config = config
                self.best_performance = performance

        return self.best_config, self.best_performance
    
    def _is_better(self, performance):
        """
        Checks if the given performance is better than the current best.
        """
        return performance < self.best_performance if self.minimize else performance > self.best_performance
    