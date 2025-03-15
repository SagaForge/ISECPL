'''
-------
tsampler.py
-------

Module responsible for sampling configurations from a given dataset. It:

    * Manages the budget through blind and observed sampling types
    * Sample filtering according to feature value


Cleaned data is saved to /temp/(dataset_name)_cleaned.csv and the path to the cleaned data is returned.

Used by: main.py

'''

import numpy as np
import pandas as pd

class Sampler:
    def __init__(self, dataset_path, budget=1000, inital_sample=0.5, performance_col=None, minimize=True, hook_func=None):
        """
        Initializes the tuning algorithm with a budget and dataset.
        
        :param budget: Total budget available for sampling.
        :param dataset: DataFrame containing the configuration data.
        """

        self.dataset = pd.read_csv(dataset_path)
        self.pure_dataset = self.dataset.loc[:, ~self.dataset.columns.str.contains('interaction')] # Dataset without interactions WITH performance
        self.features = self.pure_dataset.columns[:-1]  # Exclude last column (performance)
        self.feature_types = self._detect_feature_types()

        self.intChunkSize = 5
        self.floatChunkSize = 5
        self.feature_sample_sizes = None
        self.allocated_samples = None

        self.initial_sample = inital_sample
        self.staged = False
        self.budget_hook = hook_func

        # Tracking Stats
        self.pops = 0
        self.total_samples = 0
        self.blind_samples = 0
        self.observed_samples = 0
        self.original_size = len(self.pure_dataset)
        self.budget = budget
        self.original_budget = budget # for reporting purposes

        if performance_col:
            self.performance_col = performance_col
        else:
            # Assume last column is performance and drop 'interaction' columns
            self.performance_col = self.pure_dataset.columns[-1]

            
            #print("[tsampler.py] <INFO> No column provided, assuming last table column as performance metric, found: ", self.performance_col)


    def _detect_feature_types(self):
        """
        Detects the type of each feature (binary, integer, or float) in the dataset based on the values.
        
        :return: Dictionary where keys are feature names and values are feature types.
        """
        feature_types = {}
        
        for col in self.features:
            unique_values = self.dataset[col].dropna().unique()
            
            # If the column has only two unique values and those values are 0 and 1 (int or float)
            if set(unique_values) <= {0, 1}:  # Use <= to handle cases where only one value is present
                feature_types[col] = 'binary'
            # If the column contains only integer values (excluding floats)
            elif np.issubdtype(self.dataset[col].dtype, np.integer):
                feature_types[col] = 'integer'
            # If the column contains float values
            elif np.issubdtype(self.dataset[col].dtype, np.floating):
                feature_types[col] = 'float'
            else:
                feature_types[col] = 'unknown'  # If the feature type can't be determined
                
        return feature_types
    

    def _get_feature_sample_sizes(self):
        """
        Determines how many samples to allocate to each feature based on its type.
        
        :return: A dictionary where keys are feature names and values are the sample sizes.
        """
        feature_sample_sizes = {}
        num_features = len(self.features)
        
        # Allocate a portion of the budget for feature sampling
        feature_budget = int(self.budget * self.initial_sample)
        feature_alloc = feature_budget // num_features  # Even allocation among features initially

        if (feature_alloc <= 1):
            feature_alloc = 2
        
        # Allocate samples for each feature based on type and available budget
        for feature in self.features:
            feature_type = self.feature_types.get(feature, 'unknown')
            
            if feature_type == 'binary':
                feature_sample_sizes[feature] = feature_alloc  # All samples split between binary options
            elif feature_type == 'integer':
                max_value = self.dataset[feature].max()
                min_value = self.dataset[feature].min()
                chunk_size = 3  # Can be adjusted based on specific configuration
                num_chunks = (max_value - min_value) // chunk_size + 1
                feature_sample_sizes[feature] = feature_alloc // num_chunks
            elif feature_type == 'float':
                max_value = self.dataset[feature].max()
                min_value = self.dataset[feature].min()
                float_chunk_size = 1  # Example chunk size
                num_chunks = int((max_value - min_value) // float_chunk_size + 1)
                feature_sample_sizes[feature] = feature_alloc // num_chunks
            else:
                feature_sample_sizes[feature] = feature_alloc
        
        return feature_sample_sizes
    
    def _allocate_samples(self, feature_sample_sizes):
        """
        Allocates samples dynamically for each feature based on its data type.
        
        :param feature_sample_sizes: Dictionary with feature names and their respective sample sizes.
        :return: A dictionary structured as {feature: {range_or_value: count}}
        """
        samples_dict = {}

        for feature, sample_size in feature_sample_sizes.items():
            feature_type = self.feature_types.get(feature, 'unknown')
            samples_dict[feature] = {}

            if feature_type == 'binary':
                # For binary features, split the samples evenly between 0 and 1
                samples_dict[feature][0] = sample_size // 2
                samples_dict[feature][1] = sample_size // 2
            
            elif feature_type == 'integer':
                # For integer features, allocate based on chunks
                max_value = self.dataset[feature].max()
                min_value = self.dataset[feature].min()
                chunk_size = 1
                num_chunks = (max_value - min_value) // chunk_size + 1
                chunk_samples = sample_size // num_chunks
                
                for i in range(num_chunks):
                    lower_bound = min_value + i * chunk_size
                    upper_bound = min(min_value + (i + 1) * chunk_size - 1, max_value)
                    range_key = f"{lower_bound},{upper_bound}"
                    samples_dict[feature][range_key] = chunk_samples
            
            elif feature_type == 'float':
                # For float features, allocate based on float chunks
                max_value = self.dataset[feature].max()
                min_value = self.dataset[feature].min()
                float_chunk_size = 1  # Example chunk size
                num_chunks = int((max_value - min_value) // float_chunk_size + 1)
                chunk_samples = sample_size // num_chunks
                
                for i in range(num_chunks):
                    lower_bound = min_value + i * float_chunk_size
                    upper_bound = min_value + (i + 1) * float_chunk_size
                    range_key = f"{lower_bound:.2f},{upper_bound:.2f}"
                    samples_dict[feature][range_key] = chunk_samples
        
        return samples_dict
    

    def stage_sampler(self):
        """Check point to ensure everything is working fine, before taking samples"""
        self.feature_sample_sizes = self._get_feature_sample_sizes()
        self.allocated_samples = self._allocate_samples(self.feature_sample_sizes)

    def _budget_cost(self, val):
        if (val):
            self.budget -= val
        elif (val <= 0 and self.has_budget()):
            self.budget -= 1 # assume somewhere has rounded down the budget.
        else:
            print("[tsampler.py] <WARNING> Zero-cost sampling function called, budget remains unchanged - potential error")

    def has_budget(self):
        if (self.budget > 0):
            return True
        return False
    
    def get_performance_for_samples(self, blind_samples):
        # Ensure blind_samples is a DataFrame
        if isinstance(blind_samples, pd.Series):
            blind_samples = blind_samples.to_frame().T  # Convert Series to DataFrame

        if self.performance_col not in blind_samples.columns:
            # Merge the blind samples with the original dataset to get the performance values
            merged_samples = pd.merge(
                blind_samples,
                self.dataset,
                on=list(blind_samples.columns),  # Match on all feature columns
                how='left'
            )
            # ensure only the performance column is added
            observed_samples = merged_samples[list(blind_samples.columns) + [self.performance_col]]
            self._budget_cost(len(observed_samples)) # ensure we deduct from budget
            self.observed_samples += len(observed_samples)
            self.pure_dataset = self.pure_dataset.drop(observed_samples.index).reset_index(drop=True)
            self.pops += len(observed_samples)
            return observed_samples
        else:
            # If the performance column is already present, return the samples as-is
            return blind_samples

    def observed_random_initial_sample(self, pop=False, specific_dict=None):
        # Use the provided allocation dictionary or fall back to the default one
        allocations = self.allocated_samples if specific_dict is None else specific_dict
        sampled_rows = []
        used_indices = set()
        try:

            for feature, allocations in allocations.items():
                for value, count in allocations.items():
                    if isinstance(value, str) and ',' in value:
                        # Handle integer/float ranges
                        lower, upper = map(float, value.split(','))
                        filtered_rows = self.pure_dataset[
                            (self.pure_dataset[feature] >= lower) & (self.pure_dataset[feature] <= upper)
                        ]
                    else:
                        # Handle binary or categorical exact matches
                        filtered_rows = self.pure_dataset[self.pure_dataset[feature] == value]

                    # Sample without replacement, ensuring we don't exceed available rows
                    available_indices = list(set(filtered_rows.index) - used_indices)
                    if available_indices:
                        sampled_indices = np.random.choice(
                            available_indices, size=min(count, len(available_indices)), replace=False
                        )
                        used_indices.update(sampled_indices)
                        sampled_rows.append(self.pure_dataset.loc[sampled_indices])

                        # Remove sampled rows from dataset if pop=True
                        if pop:
                            self.pure_dataset = self.pure_dataset.drop(sampled_indices)
                            self.pops += len(sampled_indices)

            # Combine all sampled rows into a single DataFrame, dropping duplicates
            sampled_df = pd.concat(sampled_rows).drop_duplicates().reset_index(drop=True)

            if (len(sampled_df) < 1): # if it has failed for some reason, just default to a normal sample
                sampled_df = self.random_sample(self, amount=int((self.budget * self.initial_sample)), observed=True, pop=True)

            self._budget_cost(len(sampled_df))  # Update the budget based on the number of sampled rows
            self.observed_samples += len(sampled_df)
            self.total_samples += len(sampled_df)

            return sampled_df
        
        except Exception as e:
            sampled_df = self.random_sample(amount=(self.budget * self.initial_sample), observed=True, pop=True)
            self._budget_cost(len(sampled_df))
            return sampled_df

    def random_sample(self, feature=None, value=None, amount=3, observed=None, pop=True):

        rows = self.pure_dataset ## If no specific feature, sampling from entire dataset

        if feature != None: ## Targetting a specific feature
            if value != None:
                rows = self.pure_dataset[self.pure_dataset[feature] == value]
            else: ## No value provided, redundant call
                print("[tsampler.py] <ERROR> Redundant sampling function called, must provide value when sampling a specific feature.")
                return
        
        sampled_rows = rows.sample(n=min(int(amount), len(rows)), replace=False) ## Sample n rows (or the length if l < amount)

        self.total_samples += len(sampled_rows)

        if pop: ## Remove sampled data from pure dataset
            self.pure_dataset = self.pure_dataset.drop(sampled_rows.index).reset_index(drop=True)
            self.pops += len(sampled_rows)

        if observed:
            self._budget_cost(len(sampled_rows))
            sampled_df = sampled_rows.drop_duplicates().reset_index(drop=True)
            self.observed_samples += len(sampled_df)
            return sampled_df
        
        sampled_df = sampled_rows.drop(columns=[self.performance_col], errors='ignore').reset_index(drop=True) ## remove performance column
        self.blind_samples += len(sampled_df)
        
        return sampled_df
    
    def print_stats(self):
        budget_used = self.original_budget - self.budget
        print("\nSampler Report")
        print("=================")
        print(f"Total Pops: {self.pops}")
        print(f"Total Samples Given: {self.total_samples}, of which {self.blind_samples} Blind and {self.observed_samples} Observed")
        print(f"Original Pure Size:  {self.original_size}, New Size: {len(self.pure_dataset)}")
        print(f"Budget Used: {budget_used} / {self.original_budget} ({(budget_used / self.original_budget) * 100}%)")
        print("=================")
