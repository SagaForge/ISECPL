import numpy as np
import pandas as pd
from tsampler import Sampler
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


class Feature:
    """
    Feature class, acts as a node in the hierarchal tree
    """
    def __init__(self, feature, initial_samples):

        self.feature_samples = initial_samples
        self.best_value = {"value": 0, "avg_performance": 0}
        self.feature = feature

    def find_best_value():
        """"""
        ## find best value, as per self.feature_samples
        ## update self.best_value accordingly

    def add_samples(self, samples):
        """
        concats new samples into self.feature_samples, and calls _find_best_value()
        """
 
class HRASTuner:
    def __init__(self, sampler):
        self.sampler = sampler
        self.best_configuration = None
        self.best_performance = float('inf')  # Assuming lower is better
        self.feature_hierarchy = []
        self.feature_performance = {}
        self.current_samples = None ## Holds all of the current sampled configurations, including initial and 
    
    def initialize(self):
        """Performs initial sampling and constructs the feature hierarchy."""
        samples = self.sampler.observe_initial_samples()
        self.update_feature_performance(samples)
        self.build_hierarchy()
    
    def update_feature_performance(self, samples):
        """Updates the average performance for each feature-value pair."""
        performance_sums = {}
        performance_counts = {}
        
        for config, performance in samples:
            for feature, value in config.items():
                if (feature, value) not in performance_sums:
                    performance_sums[(feature, value)] = 0
                    performance_counts[(feature, value)] = 0
                performance_sums[(feature, value)] += performance
                performance_counts[(feature, value)] += 1
        
        for key in performance_sums:
            self.feature_performance[key] = performance_sums[key] / performance_counts[key]
    
    def build_hierarchy(self):
        """Constructs the feature hierarchy based on average performance."""
        sorted_features = sorted(self.feature_performance.items(), key=lambda x: x[1])
        self.feature_hierarchy = [feat for feat, _ in sorted_features]
    
    def traverse_and_sample(self):
        """Traverses the hierarchy and samples configurations accordingly."""
        while self.sampler.has_budget():
            for feature, value in self.feature_hierarchy:
                samples = self.sampler.sample_feature(feature, value)
                self.evaluate_samples(samples)
                self.update_feature_performance(samples)
                self.build_hierarchy()
    
    def evaluate_samples(self, samples):
        """Checks if any new sample is the best seen so far."""
        for config, performance in samples:
            if performance < self.best_performance:
                self.best_performance = performance
                self.best_configuration = config
    
    def run(self):
        """Executes the full HRAS tuning process."""
        self.initialize()
        self.traverse_and_sample()
        return self.best_configuration, self.best_performance


if __name__ == "__main__":
    sampler = Sampler('/home/connor/university/isecpl/temp/LLVM_cleaned.csv', budget=500, inital_sample=0.4, performance_col=None, minimize=True)
    sampler.stage_sampler() ## stage sampler for allocated samples





