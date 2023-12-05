import os
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from utils.sliding_window import sliding_window
import glob

# Function to evaluate the model
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_pred = (y_pred > 0.5).astype(int)

    # Flatten y_true and y_pred
    y = y.flatten()
    y_pred = y_pred.flatten()

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)

    # Calculate the confusion matrix
    cm = confusion_matrix(y, y_pred)

    # Convert to percentages
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create a DataFrame for the confusion matrix
    cm_df = pd.DataFrame(cm, index=[0, 1], columns=[0, 1])

    return cm_df, accuracy, precision, recall

# Function to process each CSV file
def process_file(file_path, model, window_size):
    df = pd.read_csv(file_path)
    X, y = sliding_window(df, window_size)
    return evaluate_model(model, X, y)

# Function to process all files in a directory
def test(directory, window_size):
    # Initialize an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=['filename', 'accuracy', 'precision', 'recall'])
    model_path = glob.glob('**/model-best.h5', recursive=True)[0]
    # Load the pre-trained model
    model = load_model(model_path)

    # Process each CSV file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            cm_df, accuracy, precision, recall = process_file(file_path, model, window_size)

            print(f'Filename: {filename}')
            print(f'Confusion Matrix')
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm_df, annot=True, fmt=".2%", cmap='Blues')
            plt.title('Confusion Matrix {filename}')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.savefig(f'Confusion Matrx/{filename}_confusion_matrix.png')  # Save the figure before showing it
            plt.show()
            plt.close()
            print(f'Accuracy: {accuracy}')
            print(f'Precision: {precision}')
            print(f'Recall: {recall}')

            # Append the results to the DataFrame
            results_df = results_df.append({'filename': filename, 'accuracy': accuracy, 'precision': precision, 'recall': recall}, ignore_index=True)

    # Save the results to a CSV file
    results_df.to_csv('Result/Metrics.csv', index=False)
