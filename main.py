import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import os

# Preprocessing functions (similar to Lab 1)
def remove_html(text):
    """Remove HTML tags."""
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_emoji(text):
    """Remove emojis."""
    emoji_pattern = re.compile("[" 
                               u"\U0001F600-\U0001F64F"  
                               u"\U0001F300-\U0001F5FF"  
                               u"\U0001F680-\U0001F6FF"  
                               u"\U0001F1E0-\U0001F1FF"  
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"  
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def clean_str(string):
    """Remove unwanted characters and convert to lowercase."""
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

# Function to load and preprocess the dataset
def load_and_preprocess_data(path):
    df = pd.read_csv(path)
    df = df.sample(frac=1, random_state=42)  # Shuffle the data
    
    # Clean the relevant columns
    df['cleaned_text'] = df['text'].apply(remove_html)
    df['cleaned_text'] = df['cleaned_text'].apply(remove_emoji)
    df['cleaned_text'] = df['cleaned_text'].apply(clean_str)
    
    return df

# Function to calculate the evaluation metrics
def evaluate_model(y_true, y_pred):
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mape, mae, rmse

# Path to the dataset (change to the appropriate dataset for each system/workload)
dataset_path = 'path_to_dataset.csv'

# Load and preprocess data
data = load_and_preprocess_data(dataset_path)

# Set up the independent variables (configuration values) and dependent variable (performance)
X = data.drop(columns=['performance', 'id', 'text', 'cleaned_text'])
y = data['performance']

# Initialize results list
results = []

# Repeat the training/testing process (e.g., 30 times)
REPEAT = 30
for repeat in range(REPEAT):
    # Split the dataset into 70% training and 30% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=repeat)
    
    # Initialize Linear Regression model
    model = LinearRegression()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mape, mae, rmse = evaluate_model(y_test, y_pred)
    
    # Store results
    results.append({
        'repeat': repeat + 1,
        'MAPE': mape,
        'MAE': mae,
        'RMSE': rmse
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Calculate average metrics across all repeats
avg_results = results_df.mean()

print(f"Average MAPE: {avg_results['MAPE']:.4f}%")
print(f"Average MAE: {avg_results['MAE']:.4f}")
print(f"Average RMSE: {avg_results['RMSE']:.4f}")

# Save results to CSV
output_file = 'model_results.csv'
results_df.to_csv(output_file, index=False)

print(f"Results have been saved to: {output_file}")
