import numpy as np
import pandas as pd
from tsampler import Sampler
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


class FeatureNode:
    def __init__(self, feature_name, best_value, avg_performance):
        self.feature_name = feature_name
        self.best_value = best_value
        self.avg_performance = avg_performance
        self.next = None

    def __repr__(self):
        return f"{self.feature_name}: {self.best_value} ({self.avg_performance})"

class PerformanceLinkedList:
    def __init__(self):
        self.head = None
    
    def insert_sorted(self, new_node):
        if self.head is None or new_node.avg_performance < self.head.avg_performance:
            new_node.next = self.head
            self.head = new_node
        else:
            current = self.head
            while current.next and current.next.avg_performance <= new_node.avg_performance:
                current = current.next
            new_node.next = current.next
            current.next = new_node
    
    def get_node_from_index(self, index):
        current = self.head
        count = 0
        while current:
            if count == index:
                return current
            current = current.next
            count += 1
        return None  # Return None if index is out of range
    
    def get_index_from_feature_name(self, name):
        """
        Gets the index of a node based on its feature name
        """
        current = self.head
        index = 0
        while current:
            if current.feature_name == name:
                return index
            current = current.next
            index += 1
        return -1  # Return -1 if feature name is not found
    
    def get_node_from_feature_name(self, name):
        """
        Gets a node in the linked list based on its feature name
        """
        current = self.head
        while current:
            if current.feature_name == name:
                return current
            current = current.next
        return None  # Return None if feature name is not found

    def get_list(self):
        current = self.head
        tmp = []

        while current:
            tmp.append(f'[F: {current.feature_name}, BV: {current.best_value}, AP: {int(current.avg_performance)}]')
            current = current.next

        return " ==> ".join(node for node in tmp)

class ConfigTuner:
    def __init__(self, dataset_path, budget=1000, initial_sample=0.5, performance_col=None):
        self.sampler = Sampler(dataset_path, budget, initial_sample, performance_col)
        self.sampler.stage_sampler() ## Set up sampler

        self.linked_list = PerformanceLinkedList()
        self.sampled_data = None

        # Config tracking
        self.best_config = None
        self.best_performance = 1000000000000000000 # Some Large Number
        self.current_feature = None

        # Stats Tracking
        self.hierarchy_changes = 0
        self.highest_performance_feature = None
        self.lowest_performance_feature = None
    
    def _compute_performance_hierarchy(self):

        feature_performance = {}
        performance_hierarchy = PerformanceLinkedList()

        sampled_data = self.sampled_data
        feature_count = -1

        for feature in self.sampler.features:
            feature_count += 1 ## update index of our feature
            values = sampled_data[feature].unique()
            feature_performance[feature] = {}
            
            for value in values:
                subset = sampled_data[sampled_data[feature] == value]
                avg_perf = subset[self.sampler.performance_col].mean()
                feature_performance[feature][value] = avg_perf
                
            best_value = min(feature_performance[feature], key=feature_performance[feature].get)
            best_avg_perf = feature_performance[feature][best_value]
            
            node = FeatureNode(feature, best_value, best_avg_perf)
            performance_hierarchy.insert_sorted(node)

        # Go through the current linked list, if a feature hierarchy has changed in our new list, 
        # return head of new list, else return next feature of new list to explore
        current = self.linked_list.head
        count = 0
        while current:
            parallel_node = performance_hierarchy.get_node_from_index(count)
            if (current.feature_name != parallel_node.feature_name):
                # if feature hierarchy has changed, return first node that has experienced a hierarchy change
                print(f"[tuner.py] <INFO> Hierarchy change occured, original feature {current.feature_name}, ~{int(current.avg_performance)} swapped for {parallel_node.feature_name}, ~{int(parallel_node.avg_performance)}")
                self.linked_list = performance_hierarchy
                return performance_hierarchy.head # just return head for now
            count += 1
            current = current.next


        self.linked_list = performance_hierarchy

        if self.current_feature == None:
            return self.linked_list.head
        # else, return the next feature to explore in the new hierarchy
        
        return self.linked_list.get_node_from_feature_name(self.current_feature.feature_name).next
    
    def _find_best_config(self, sampled_data):
        """
        Compares the current best configuration with new sampled data and updates if a better configuration is found.
        """
        for _, row in sampled_data.iterrows():
            performance = row[self.sampler.performance_col]
            if performance < self.best_performance:
                self.best_performance = performance
                self.best_config = row.drop(labels=[self.sampler.performance_col]).to_dict()

    def _update_performance_data(self, sampled_data):

        self.sampled_data = pd.concat([self.sampled_data, sampled_data]).drop_duplicates()
        return sampled_data ## returns specific sampled_data if required

    def _explore_feature_space(self, feature_node):
        name = feature_node.feature_name
        bv = feature_node.best_value

        est_size = int(self.sampler.budget*0.05)

        if (est_size <= 2):
            est_size = 5

        if (self.sampler.budget - est_size < 0):
            est_size = self.sampler.budget

        sampled_data = self._update_performance_data(
            self.sampler.random_sample(feature=name, value=bv, amount=est_size, observed=True, pop=True))
        
        return sampled_data

    def tuning_report(self):
        print("\n\n<--------------------------> TUNING COMPLETE <------------------------>")
        print("\n\n<< Configuration Tuning Report >>")
        print("==========================================================")
        print("Configurations")
        print(f"Best Configuration:         {self.best_config}")
        print(f"Best Configuration Performance:         {self.best_performance}")
        print(f"Feature Hierarchy:          {self.linked_list.get_list()}")
        print("\nStatistics")
        print(f"Runtime: ")
        print(f"Total Samples Recieved: {len(self.sampled_data)}")
        print("==========================================================")


    def run(self):
        """"""
        ### Initial Sampling ###
        initial_sample = self.sampler.observed_random_initial_sample(pop=True)
        self._update_performance_data(initial_sample)
        self.current_feature = self._compute_performance_hierarchy() # calculate hierarchy

        while self.sampler.has_budget(): ## Feature Specific Sampling, Working Down the Hierarchy
            print("\n\n[tuner.py] <INFO> Feature Exploration Step: ", self.current_feature.feature_name)
            feature_samples = self._explore_feature_space(self.current_feature)
            self._find_best_config(feature_samples)
            self.current_feature = self._compute_performance_hierarchy()

        self.tuning_report()
        self.sampler.print_stats()


if __name__ == "__main__":
    ### Settings ###
    dataset_path = '/home/connor/university/isecpl/temp/LLVM_cleaned.csv'
    budget = 5000
    initial_sample = 0.2
    performance_col = None
    minimize = True

    tuner = ConfigTuner(dataset_path, budget=budget, initial_sample=initial_sample, performance_col=performance_col)
    tuner.run()






