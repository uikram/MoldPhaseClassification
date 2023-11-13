import os
import pandas as pd
import shutil

# Function to categorize phase
def categorize_phase(phase):
    if phase < 0.25:
        return 1
    elif 0.25 <= phase < 0.50:
        return 2
    elif 0.50 <= phase < 0.75:
        return 3
    elif 0.75 <= phase:
        return 4

def label_data(input_directory):
    output_directory = 'Processed/Labelled_Data'
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Traverse subfolders in 'Raw_Data'
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
                df['HOUR'] = pd.to_datetime(df['HOUR'], format='%Y%m%d%H')
                df = df.sort_values('HOUR')
                df = df[['HOUR','MOLD_ID', 'CT', 'TAV', 'SHOT_COUNT', 'PHASE']]
                df.fillna(0, inplace=True)
                
                # Save the modified DataFrame with the original file name in 'Labelled_Data'
                output_file_path = os.path.join(output_directory, file)
                df.to_csv(output_file_path, index=False)
                
                print(f"Processed: {file_path}")


#Extracts Data that contains phases from 1 to 4
def all_phase(input_folder):
    # Set the path to the folder where you want to copy the selected CSV files
    output_folder = 'Processed/Labelled_Data_Complete_Phase(1-4)/'
        # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # create an empty list to store the filenames
    files_with_all_phases = []

    # iterate over all files in the source directory
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):  # check if the file is a CSV
            file_path = os.path.join(input_folder, filename)
            df = pd.read_csv(file_path)  # read the CSV file into a DataFrame

            # check if 'PHASE' column exists and its unique values contain 1, 2, 3, and 4
            if 'PHASE' in df.columns and set([1, 2, 3, 4]).issubset(df['PHASE'].unique()):
                files_with_all_phases.append(filename)
                # copy the file to the destination directory
                shutil.copy(file_path, output_folder)

    # print out the filenames
    print("The following files contain all four phases:")
    for filename in files_with_all_phases:
        print(filename)