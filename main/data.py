'''
-------
data.py
-------

Data cleaning pipeline used prior to performance tuning. It leverages:

    * Standard Regex Text Cleaning
    * Outlier Detection & Removal (Using Z-Score + IQR)
    * Imputation of Missing Values using Column Mean Average
    * Automated Feature Interaction Generation (On a A x B basis per permutation)
    * Logarithmic Performance Scaling

Cleaned data is saved to /temp/(dataset_name)_cleaned.csv and the path to the cleaned data is returned.

Used by: main.py

'''

import pandas as pd
import numpy as np
import re
from scipy import stats
from sklearn.impute import SimpleImputer
from itertools import combinations
import os

class DataCleaner:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.original_df = pd.read_csv(dataset_path)
        self.cleaned_df = self.original_df.copy()
        
        # Initialize counters for various cleaning actions
        self.outliers_removed = 0
        self.imputed_values = 0
        self.removed_data = 0
        self.feature_interactions = 0

    def clean_symbols_and_texts(self):
        """Remove non-interpretable symbols, emojis, and unicode characters."""
        def clean_text(text):
            # Remove non-ASCII characters and special symbols using regex
            text = re.sub(r'[^\x00-\x7F]+', ' ', str(text))
            # Remove emojis and non-sensical characters
            text = re.sub(r'[^\w\s]', '', text)
            return text

        # Apply cleaning to all text columns (columns with object data type)
        for col in self.cleaned_df.select_dtypes(include=['object']).columns:
            self.cleaned_df[col] = self.cleaned_df[col].apply(clean_text)
        return self.cleaned_df

    def detect_outliers(self):
        """Detect outliers using two statistical tests: Z-Score and IQR, with more conservative thresholds."""
        numeric_cols = self.cleaned_df.select_dtypes(include=[np.number]).columns

        # Z-Score method with increased threshold
        z_scores = np.abs(stats.zscore(self.cleaned_df[numeric_cols], nan_policy='omit'))
        z_outliers = (z_scores > 6).any(axis=1)  # Increased threshold to 6

        # IQR method with increased multiplier for IQR
        Q1 = self.cleaned_df[numeric_cols].quantile(0.10)
        Q3 = self.cleaned_df[numeric_cols].quantile(0.90)
        IQR = Q3 - Q1
        iqr_outliers = ((self.cleaned_df[numeric_cols] < (Q1 - 3 * IQR)) |  # Increased multiplier to 3
                        (self.cleaned_df[numeric_cols] > (Q3 + 3 * IQR))).any(axis=1)

        # Combine outliers from both methods
        outliers = z_outliers | iqr_outliers

        # Count rows before and after outlier removal
        rows_before = self.cleaned_df.shape[0]
        self.cleaned_df = self.cleaned_df[~outliers]
        rows_after = self.cleaned_df.shape[0]

        # Update the count of outliers removed
        outliers_removed = rows_before - rows_after
        self.outliers_removed += outliers_removed

        return self.cleaned_df

    def impute_missing_values(self):
        """Impute missing values using the column mean."""
        numeric_cols = self.cleaned_df.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='mean')
        self.cleaned_df[numeric_cols] = imputer.fit_transform(self.cleaned_df[numeric_cols])

        # Count imputed values
        imputed_data = self.cleaned_df.isna().sum().sum()
        self.imputed_values += imputed_data
        return self.cleaned_df

    def generate_feature_interactions(self):
        """Generate feature interactions between numerical columns"""
        numeric_cols = self.cleaned_df.select_dtypes(include=[np.number]).columns
        interaction_data = {}  # Dictionary to store named interaction columns

        for (col1, col2) in combinations(numeric_cols, 2):
            interaction_col = f"{col1}_{col2}_interaction"
            ## feature interactions generated on simple A x B basis (interaction analysis not performed here)
            interaction_data[interaction_col] = self.cleaned_df[col1] * self.cleaned_df[col2]

        # Concatenate all interaction columns at once with proper column names
        self.cleaned_df = pd.concat([self.cleaned_df, pd.DataFrame(interaction_data)], axis=1)
        
        self.feature_interactions = len(list(combinations(numeric_cols, 2)))
        return self.cleaned_df

    def save_cleaned_data(self):
        """Save the cleaned dataset to the specified directory."""
        # Get the cleaned file path
        base_name = os.path.basename(self.dataset_path)
        name, ext = os.path.splitext(base_name)
        cleaned_path = os.path.join('/home/connor/university/isecpl/temp/', f"{name}_cleaned{ext}")
        self.cleaned_df.to_csv(cleaned_path, index=False)
        return cleaned_path
    
    def clean_data(self):
        """Run the entire cleaning process and return summary."""
        self.clean_symbols_and_texts()
        self.detect_outliers()
        self.impute_missing_values()
        self.generate_feature_interactions()
        cleaned_path = self.save_cleaned_data()

        # Output summary
        print("\n[data.py] <INFO> Cleaning Report:")
        print(f"            Total Data Removed: {self.removed_data + self.outliers_removed} rows")
        print(f"            Imputed Values: {self.imputed_values} values")
        print(f"            Outliers Removed: {self.outliers_removed} rows")
        print(f"            Feature Interaction Generations: {self.feature_interactions} interactions")
        print(f"            Cleaned dataset saved as: {cleaned_path}\n")
        return cleaned_path

