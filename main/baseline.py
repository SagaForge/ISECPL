'''
-------
baseline.py
-------

Module for implementing a standard Random Search baseline for CPT prior to tuning:

    * Random Search Baseline, given some budget b

Returns the entire, best configurgation found by Random Search. 
    
Used by: main.py

'''

import pandas as pd
import numpy as np

class RandomSearchBaseline:
    def __init__(self, dataset_path, budget, minimize=True):
        """
        Initializes the Random Search Baseline Tool.
        :param dataset_path: Path to the dataset (CSV format assumed)
        :param budget: Maximum number of measurements allowed
        :param minimize: Boolean indicating whether to minimize or maximize performance
        """
        self.dataset = pd.read_csv(dataset_path)
        self.budget = min(budget, len(self.dataset))  # Ensure budget does not exceed dataset size
        self.minimize = minimize
        self.best_config = None
        self.best_performance = np.inf if minimize else -np.inf

    def random_search(self):
        """
        Performs Random Search within the allocated budget.
        """
        sampled_indices = np.random.choice(self.dataset.index, self.budget, replace=False)
        
        for idx in sampled_indices:
            config = self.dataset.loc[idx, self.dataset.columns[:-1]].to_dict()
            performance = self.dataset.loc[idx, self.dataset.columns[-1]]
            
            if self._is_better(performance):
                self.best_config = config
                self.best_performance = performance
        
        return self.best_config, self.best_performance
    
    def _is_better(self, performance):
        """
        Checks if the given performance is better than the current best.
        """
        if self.minimize:
            return performance < self.best_performance
        else:
            return performance > self.best_performance
