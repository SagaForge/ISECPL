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

        self.dataset = pd.read_csv(dataset_path)
        self.budget = int(min(budget, len(self.dataset)))
        self.minimize = minimize

        self.dataset = self.dataset.loc[:, ~self.dataset.columns.str.contains('interaction')]

        if performance_col:
            self.performance_col = performance_col
        else:
            
            self.performance_col = self.dataset.columns[-1]

        self.best_config = None
        self.best_performance = np.inf if minimize else -np.inf
        
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
    