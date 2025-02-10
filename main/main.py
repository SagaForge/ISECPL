# PACKAGE IMPORTS #

# LOCAL IMPORTS #
from data import DataCleaner ## data.py, cleans data prior to CPT
from baseline import RandomSearchBaseline ## baseline.py, for Random Search baseline


pathToDataset = ""

def main():
    # Specify the path to the dataset
    dataset_path = '/home/connor/university/isecpl/datasets/LLVM.csv'
    
    # Create an instance of DataCleaner
    cleaner = DataCleaner(dataset_path)
    
    # Run the data cleaning process
    cleaned_data = cleaner.clean_data()

    baseliner = RandomSearchBaseline('/home/connor/university/isecpl/temp/LLVM_cleaned.csv', 1000, performance_col=None, minimize=True)

    result = baseliner.random_search()

    print("\n\nRandom Search Baseline Performance: ", result[1], " for Configuration: ", result[0])


if __name__ == "__main__":
    main()