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
    def __init__(self, dataset_path, budget, performance_col=None, minimize=True):
        """
        Initializes the tuning algorithm with a budget and dataset.
        
        :param budget: Total budget available for sampling.
        :param dataset: DataFrame containing the configuration data.
        """
        self.budget = budget
        self.dataset = pd.read_csv(dataset_path)
        self.pureDataset = self.dataset.loc[:, ~self.dataset.columns.str.contains('interaction')] # Dataset without interactions WITH performance
        self.features = self.pureDataset.columns[:-1]  # Exclude last column (performance)
        self.feature_types = self._detect_feature_types()

        self.intChunkSize = 5
        self.floatChunkSize = 5
        self.feature_sample_sizes = None
        self.allocted_samples = None

        if performance_col:
            self.performance_col = performance_col
        else:
            # Assume last column is performance and drop 'interaction' columns
            self.performance_col = self.pureDataset.columns[-1]

            print("[tsampler.py] <INFO> No column provided, assuming last table column as performance metric, found: ", self.performance_col)


    def _detect_feature_types(self):
            """
            Detects the type of each feature (binary, integer, or float) in the dataset based on the values.
            
            :return: Dictionary where keys are feature names and values are feature types.
            """
            feature_types = {}
            
            for col in self.features:
                unique_values = self.dataset[col].dropna().unique()
                
                # If the column has only two unique values and those values are 0 and 1 (int or float)
                if set(unique_values) == {0, 1}:
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
        remaining_budget = self.budget
        num_features = len(self.features)
        
        # Allocate 500 budget (as per your example) for feature sampling
        feature_budget = 500
        feature_alloc = feature_budget // num_features  # Even allocation among features initially
        
        # Allocate samples for each feature based on type and available budget
        for feature in self.features:
            feature_type = self.feature_types.get(feature, 'unknown')
            #print(feature_type)
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
                # Here we might define chunk size for float values
                float_chunk_size = 1  # Example chunk size
                num_chunks = int((max_value - min_value) // float_chunk_size + 1)
                feature_sample_sizes[feature] = feature_alloc // num_chunks
            else:
                feature_sample_sizes[feature] = feature_alloc
        
        #print(feature_sample_sizes)
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
                    chunk_size = 3
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
                    float_chunk_size = 1
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
        self.allocted_samples = self._allocate_samples(self.feature_sample_sizes)

        print("[tsampler.py] <INFO> Sampling allocation complete, ready to sample using (feature: {allocations}): ", self.allocted_samples)
    

    def blind_random_sample(self, pop=False, specific_dict=None):
        allocations = self.allocted_samples if specific_dict is None else specific_dict
        sampled_rows = []
        used_indices = set()

        for feature, allocations in allocations.items():
            for value, count in allocations.items():
                if isinstance(value, str) and ',' in value:
                    # Handle integer/float ranges
                    lower, upper = map(float, value.split(','))
                    filtered_rows = self.pureDataset[
                        (self.pureDataset[feature] >= lower) & (self.pureDataset[feature] <= upper)
                    ]
                else:
                    # Handle binary or categorical exact matches
                    filtered_rows = self.pureDataset[self.pureDataset[feature] == value]

                # Sample without replacement, ensuring we don't exceed available rows
                available_indices = list(set(filtered_rows.index) - used_indices)
                if available_indices:
                    sampled_indices = np.random.choice(available_indices, size=min(count, len(available_indices)), replace=False)
                    used_indices.update(sampled_indices)
                    sampled_rows.append(self.pureDataset.loc[sampled_indices])

                    # Remove sampled rows from dataset if pop=True
                    if pop:
                        self.pureDataset = self.pureDataset.drop(sampled_indices)

        # Combine all sampled rows into a single DataFrame, dropping the performance column (blind)
        sampled_df = pd.concat(sampled_rows).drop(columns=[self.performance_col], errors='ignore').drop_duplicates().reset_index(drop=True)
        return sampled_df
    
    def _budget_cost(self, val):
        if (val):
            self.budget -= val
            print("[tsampler.py] <INFO> Observed sampling function called, deducting ", val, " from budget. New budget: ", self.budget)
        else:
            print("[tsampler.py] <WARNING> Zero-cost sampling function called, budget remains unchanged - potential error")

    def observed_random_sample(self, pop=False, specific_dict=None):
        allocations = self.allocted_samples if specific_dict is None else specific_dict
        sampled_rows = []
        used_indices = set()

        for feature, allocations in allocations.items():
            for value, count in allocations.items():
                if isinstance(value, str) and ',' in value:
                    # Handle integer/float ranges
                    lower, upper = map(float, value.split(','))
                    filtered_rows = self.pureDataset[
                        (self.pureDataset[feature] >= lower) & (self.pureDataset[feature] <= upper)
                    ]
                else:
                    # Handle binary or categorical exact matches
                    filtered_rows = self.pureDataset[self.pureDataset[feature] == value]

                # Sample without replacement, ensuring we don't exceed available rows
                available_indices = list(set(filtered_rows.index) - used_indices)
                if available_indices:
                    sampled_indices = np.random.choice(available_indices, size=min(count, len(available_indices)), replace=False)
                    used_indices.update(sampled_indices)
                    sampled_rows.append(self.pureDataset.loc[sampled_indices])

                    # Remove sampled rows from dataset if pop=True
                    if pop:
                        self.pureDataset = self.pureDataset.drop(sampled_indices)

        # Combine all sampled rows into a single DataFrame, dropping the performance column (blind)
        sampled_df = pd.concat(sampled_rows).drop_duplicates().reset_index(drop=True)
        self._budget_cost(len(sampled_df))
        return sampled_df


    def observed_feature_sample(self, feature_dict):
        """Some code here, to be implemented"""
   
if __name__ == "__main__":

    sampler = Sampler('/home/connor/university/isecpl/datasets/LLVM.csv', 1000, performance_col=None, minimize=True)

    a = sampler.stage_sampler()
    #b = sampler.blind_random_sample()
    b = sampler.observed_random_sample()

    #print(b)

    #print(a)
    