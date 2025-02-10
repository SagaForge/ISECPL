from data import DataCleaner ## data.py, cleans data prior to CPT

def main():
    # Specify the path to the dataset
    dataset_path = '/home/connor/university/isecpl/datasets/7z.csv'
    
    # Create an instance of DataCleaner
    cleaner = DataCleaner(dataset_path)
    
    # Run the data cleaning process
    cleaned_data = cleaner.clean_data()

if __name__ == "__main__":
    main()