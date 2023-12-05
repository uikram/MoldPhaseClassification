import os
import pandas as pd
import shutil
import numpy as np

# Function to categorize phase
def categorize_phase(phase):
    if phase < 0.50:
        return 0
    elif phase <= 50:
        return 1


def label_data(input_directory):
    output_directory = 'Processed/Labelled_Data'
    
    # Traverse subfolders in input_directory
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith("_statistics.csv"):
                # Construct the full path to the CSV file
                file_path = os.path.join(root, file)
                
                # Load the CSV file into a DataFrame
                df = pd.read_csv(file_path)
                
                # Check if DataFrame is empty or only contains the specified columns
                if df.empty or set(df.columns) == {'TFF', 'CT', 'TAV', 'SHOT_COUNT'}:
                    print(f'File {file} is empty')
                    continue
                
                # Apply your transformations
                df = df[df['TFF'] > 20180000000000]
                df = df[(df['CT'] != 9999) & (df['CT'] != 999.9)]
                df['LIFE_CYCLE'] = df['SC'] / df['max_shots']
                df['PHASE'] = df['LIFE_CYCLE'].apply(categorize_phase)
                df['TFF'] = df['TFF'].astype(str)
                df['HOUR'] = df['TFF'].str[:10]

                df = df[df['SHOT_COUNT'] >= 0]
                df_non_zero = df[df['SHOT_COUNT'] > 0]

                Q1 = df_non_zero['SHOT_COUNT'].quantile(0.25)
                Q3 = df_non_zero['SHOT_COUNT'].quantile(0.75)
                IQR = Q3 - Q1

                filter = (df['SHOT_COUNT'] <= Q3 + 0.75 * IQR) 
                df = df[['HOUR','MOLD_ID', 'CT', 'TAV', 'SHOT_COUNT', 'PHASE']]
                df = df.loc[filter]  
                df.fillna(0, inplace=True)
                
                df['HOUR'] = pd.to_datetime(df['HOUR'], format='%Y%m%d%H')
                df = df.sort_values('HOUR')
                # Create the same subfolder structure in output_directory
                relative_path = os.path.relpath(root, input_directory)
                output_subfolder = os.path.join(output_directory, relative_path)
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)
                
                # Save the modified DataFrame with the original file name in the corresponding subfolder
                output_file_path = os.path.join(output_subfolder, file)
                df.to_csv(output_file_path, index=False)
                
                print(f"Processed: {file_path}")


#Extracts Data that contains phases from 1 to 4
def all_phase_sort(input_directory):
    output_directory = 'Processed/Labelled_Data_Complete_Phase(0-1)'
    
    # Traverse subfolders in input_directory
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith('.csv'):  # check if the file is a CSV
                # Construct the full path to the CSV file
                file_path = os.path.join(root, file)
                
                # Load the CSV file into a DataFrame
                df = pd.read_csv(file_path)
                
                # Check if 'PHASE' column exists and its unique values contain 1, 2, 3, and 4
                if 'PHASE' in df.columns and set([0, 1]).issubset(df['PHASE'].unique()):
                    # Create the same subfolder structure in output_directory
                    relative_path = os.path.relpath(root, input_directory)
                    output_subfolder = os.path.join(output_directory, relative_path)
                    if not os.path.exists(output_subfolder):
                        os.makedirs(output_subfolder)
                    
                    # Copy the file to the corresponding subfolder in output_directory
                    output_file_path = os.path.join(output_subfolder, file)
                    shutil.copy(file_path, output_file_path)
                    
                    print(f"Copied: {file_path} to {output_file_path}")