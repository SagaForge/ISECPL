import numpy as np
import pandas as pd
from tsampler import Sampler
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import time
import traceback
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


class FeatureNode:
    def __init__(self, feature_name, best_value, avg_performance):
        self.feature_name = feature_name
        self.best_value = best_value
        self.avg_performance = avg_performance
        self.index = 0
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

    def get_list_string(self):
        current = self.head
        tmp = []

        while current:
            tmp.append(f'[F: {current.feature_name}, BV: {current.best_value}, AP: {int(current.avg_performance)}]')
            current = current.next

        return " ==> ".join(node for node in tmp)
    
    def get_hierarchy_feature_only_string(self):
        current = self.head
        tmp = []

        while current:
            tmp.append(f'[{current.feature_name}]')
            current = current.next

        return "==>".join(node for node in tmp)

class ConfigTuner:
    def __init__(self, dataset_path, budget=1000, initial_sample=0.5, performance_col=None):
        self.sampler = Sampler(dataset_path, budget=budget, inital_sample=initial_sample, performance_col=performance_col)
        self.sampler.stage_sampler() ## Set up sampler

        self.linked_list = PerformanceLinkedList()
        self.sampled_data = None

        # Config tracking
        self.best_config = None
        self.best_performance = 1000000000000000000000000000000000000000000000 # Some Unnaturally Large Number
        self.current_feature = None

        # Stats Tracking
        self.hierarchy_changes = 0
        self.highest_performance_feature = None
        self.lowest_performance_feature = None


        # Gaussian Essentials
        #self.gp = GaussianProcessRegressor(kernel=ConstantKernel(1.0) * RBF(length_scale=1.0), alpha=1e-5)
        # Increase max_iter in the GaussianProcessRegressor
        self.gp = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * RBF(length_scale=1.0),
            n_restarts_optimizer=10,  # Increase the number of restarts
            optimizer='fmin_l_bfgs_b',  # Use L-BFGS-B optimizer
            random_state=42,
            alpha=1e-5,  # Add a small noise term to improve stability
            normalize_y=True  # Normalize the target variable (performance)
            #max_iter=1000  # Increase the maximum number of iterations
        )



    def _next_node_on_hierarchy_change(self, performance_hierarchy, current, parallel_node):

        current_feature_rank = self.linked_list.get_index_from_feature_name(self.current_feature.feature_name)
        current_rank = self.linked_list.get_index_from_feature_name(current.feature_name)
        parallel_rank = performance_hierarchy.get_index_from_feature_name(parallel_node.feature_name)

        next_node = None

        #print(f"Ranks, CFR: {current_feature_rank} CR: {current_rank} PR: {parallel_rank}")

        if (current_feature_rank < parallel_rank): # if we havent reached that feature in exploration yet, continue to next feature
            next_node = performance_hierarchy.get_node_from_index(current_feature_rank + 1)
            #print(f"[tuner.py] <INFO> Hierarchy change occured, but current feature outranks. Updating hierarchy and continuing to next node: {self.current_feature.feature_name} -> {next_node.feature_name}")
        elif (current_feature_rank > current_rank): # hierarchy change has occured at a higher-ranking feature, revist and explore
            next_node = performance_hierarchy.get_node_from_index(parallel_rank)
            #print(f"[tuner.py] <INFO> Hierarchy change occured at higher-ranking feature, revisiting this node and re-exploring: {self.current_feature.feature_name} -> {next_node.feature_name}")
        elif (current_feature_rank == parallel_rank):
            next_node = performance_hierarchy.get_node_from_index(parallel_rank)
        
        self.linked_list = performance_hierarchy
        return next_node

    def _compute_performance_hierarchy(self):

        feature_performance = {}
        parallel_hierarchy = PerformanceLinkedList()

        sampled_data = self.sampled_data
        feature_count = -1

        for feature in self.sampler.features:
            feature_count += 1 ## update index of our feature
            values = sampled_data[feature].unique()
            feature_performance[feature] = {}
            
            for value in values:
                subset = sampled_data[sampled_data[feature] == value]
                avg_perf = float(subset[self.sampler.performance_col].mean())
                feature_performance[feature][value] = float(avg_perf)

            #print("F", feature)
            #print(self.sampler.feature_types)
            #print(self.sampler.allocated_samples)
            #print(sampled_data[feature])
                
            best_value = min(feature_performance[feature], key=feature_performance[feature].get)
            best_avg_perf = feature_performance[feature][best_value]
            
            node = FeatureNode(feature, best_value, best_avg_perf)
            parallel_hierarchy.insert_sorted(node)

        # Go through the current linked list, if a feature hierarchy has changed in our new list
        current = self.linked_list.head
        count = 0
        while current:
            parallel_node = parallel_hierarchy.get_node_from_index(count)
            if (current.feature_name != parallel_node.feature_name):
                # Hierarchy change has occured, find the next node to visit based on hierarchy logic
                avg_change = ((parallel_node.avg_performance - current.avg_performance) / parallel_node.avg_performance) * 100
                #print("<Percentage Performance Change>: ", avg_change, "%")
                return self._next_node_on_hierarchy_change(parallel_hierarchy, current, parallel_node)    
            count += 1
            current = current.next

        self.linked_list = parallel_hierarchy

        if self.current_feature == None:
            return self.linked_list.head
        
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

    def _explore_feature_space(self, feature_node, observed=True, pop=True):
        name = feature_node.feature_name
        bv = feature_node.best_value

        est_size = int(self.sampler.budget*0.10)

        if (est_size <= 2):
            est_size = 5

        if (self.sampler.budget - est_size < 0):
            est_size = self.sampler.budget

        sampled_data = self._update_performance_data(
            self.sampler.random_sample(feature=name, value=bv, amount=est_size, observed=observed, pop=pop))
        
        return sampled_data

    def tuning_report(self):
        print("\n\n<--------------------------> TUNING COMPLETE <------------------------>")
        print("\n\n<< Configuration Tuning Report >>")
        print("==========================================================")
        print(f"Best Configuration:         {self.best_config}")
        print(f"Best Configuration Performance:         {self.best_performance}")
        print(f"Feature Hierarchy:          {self.linked_list.get_list_string()}")
        print("\nStatistics")
        print(f"Runtime: ")
        print(f"Total Samples Recieved: {len(self.sampled_data)}")
        print("==========================================================")

    
    def _hras_standard(self):
        """
        HRAS Standard

        Constructs feature hierarchy from initial sample, then
        """
        initial_sample = self.sampler.observed_random_initial_sample(pop=True)
        self._update_performance_data(initial_sample)
        self.current_feature = self._compute_performance_hierarchy() # calculate hierarchy

        while self.sampler.has_budget():
            if (self.current_feature == None):
                self.current_feature = self.linked_list.get_node_from_index(0)

            feature_samples = self._explore_feature_space(self.current_feature)
            self._find_best_config(feature_samples)
            self.current_feature = self.current_feature.next

        #self.tuning_report()
        #self.sampler.print_stats()


    def _hras_adaptive(self, explore=False):
        """
        Adaptive HRAS that updates and navigates feature hierarchy as new samples come in

        Explorative option, which dedicates some budget to random exploration at regardless of features

        """
        initial_sample = self.sampler.observed_random_initial_sample(pop=True)
        self._update_performance_data(initial_sample)
        self.current_feature = self._compute_performance_hierarchy() # calculate hierarchy

        while self.sampler.has_budget(): ## Feature Specific Sampling, Working Down the Hierarchy
            if (self.current_feature == None): # likely due to end of hierarchy, so just continue sampling from the start of the hierarchy
                self.current_feature = self.linked_list.get_node_from_index(0)

            feature_samples = self._explore_feature_space(self.current_feature)
            self._find_best_config(feature_samples)


            if (explore and self.sampler.has_budget()): # if random exploration enabled
                sample_size = int(self.sampler.budget * 0.02)
                if (sample_size <= 0):
                    sample_size = 1
                addit_samples = self.sampler.random_sample(amount=sample_size, observed=True, pop=True)
                self._find_best_config(addit_samples)
                self._update_performance_data(addit_samples)

            self.current_feature = self._compute_performance_hierarchy()

        #self.tuning_report()
        #self.sampler.print_stats()


    def _hras_gaussian(self):
        """
        Adaptive HRAS that uses Gaussian processes to estimate the performance of samples
        and guide the sampling process towards successful samples (minimizing performance)
        while adapting to feature importance
        """

        # Initial random sampling to bootstrap the GP and construct feature hierarchy
        initial_sample = self.sampler.observed_random_initial_sample(pop=True)
        self._update_performance_data(initial_sample)
        self.current_feature = self._compute_performance_hierarchy()  # compute feature hierarchy and get best performing feature

        # Fit the GP on the initial samples
        X = initial_sample.drop(columns=[self.sampler.performance_col]).values
        y = initial_sample[self.sampler.performance_col].values
        self.gp.fit(X, y)

        self.gp.kernel_.k1.constant_value_bounds = (1e-20, 1e20)

        while self.sampler.has_budget():
            print(self.sampler.budget)
            if self.current_feature is None:
                self.current_feature = self.linked_list.get_node_from_index(0)

            # Blind Sample within the current feature space
            blind_feature_samples = pd.DataFrame()
            for i in range(0, 3, 1):  # get three lots of random blind samples
                blind_feature_samples = pd.concat(
                    [self._explore_feature_space(self.current_feature, observed=False, pop=False),
                    blind_feature_samples], ignore_index=True
                )

            # Predict performance using the GP
            X_candidates = blind_feature_samples.values
            y_pred, y_std = self.gp.predict(X_candidates, return_std=True)

            # Select the top 25% of samples with the lowest predicted performance
            num_samples = len(y_pred)
            top_25_percent = int(np.ceil(0.25 * num_samples))
            best_sample_indices = np.argpartition(y_pred, top_25_percent)[:top_25_percent]

            # Observe the performance of the top 25% samples
            best_samples = pd.DataFrame()
            for idx in best_sample_indices:
                best_sample = blind_feature_samples.iloc[idx]
                observed_best_sample = self.sampler.get_performance_for_samples(best_sample)

                # Update the GP with the new observation
                X = np.vstack((X, X_candidates[idx]))
                y = np.append(y, observed_best_sample[self.sampler.performance_col])
                self.gp.fit(X, y)

                best_samples = pd.concat([observed_best_sample, best_samples], ignore_index=True)

            self._find_best_config(best_samples)        # get best config from observed best samples
            self._update_performance_data(best_samples) # update performance data with new observed samples

            # Compute and move to the next feature in the hierarchy
            self.current_feature = self._compute_performance_hierarchy()

    def run(self, mode=1):

        try:
            if mode == 1: # hras, non-adaptive
                self._hras_standard()

            elif mode == 2: # hras, adaptive
                self._hras_adaptive()

            elif mode == 3: #hras, adaptive and explorative
                self._hras_adaptive(explore=True)

            elif mode == 4: # hras, adaptive with gaussian prediction
                self._hras_gaussian()

        except Exception as e:

            traceback.print_exc()



if __name__ == "__main__":
    tuner = ConfigTuner(dataset_path="/home/connor/university/isecpl/temp/LLVM_cleaned.csv", budget=500, initial_sample=0.2, performance_col=None)

    tuner.run(4)

    tuner.tuning_report() # print results

    tuner.sampler.print_stats()