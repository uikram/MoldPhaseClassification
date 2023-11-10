import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def filter(dataframe):
    # Define the columns you want to keep
    selected_columns = ['HOUR', 'TAV', 'SHOT_COUNT', 'CT', 'PHASE','MOLD_ID']
    
    # Filter the DataFrame to keep only the selected columns
    filtered_df = dataframe[selected_columns]
    
    return filtered_df

# Function to convert selected columns to numpy arrays
def features(dataframe):
    # Extract TAV, SHOT_COUNT, and PHASE as numpy arrays
    tav_array = dataframe['TAV'].to_numpy()
    shot_count_array = dataframe['SHOT_COUNT'].to_numpy()
    ct_array = dataframe['CT'].to_numpy()
    
    return tav_array, shot_count_array, ct_array


# def plots(df, output_folder='Plots'):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
    
#     tav_array, shot_count_array, ct_array = features(df)
#     mold_id = df['MOLD ID'].unique
    
#     # Create subplots
#     fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=['TAV', 'SHOT_COUNT', 'CT'])
    
#     # Add traces to subplots
#     for i in range(len(tav_array)):
#         fig.add_trace(go.Scatter(x=df['HOUR'], y=tav_array[i], mode='lines', name=f'TAV ({mold_id[i]})'), row=1, col=1)
#         fig.add_trace(go.Scatter(x=df['HOUR'], y=shot_count_array[i], mode='lines', name=f'SHOT_COUNT ({mold_id[i]})'), row=2, col=1)
#         fig.add_trace(go.Scatter(x=df['HOUR'], y=ct_array[i], mode='lines', name=f'CT ({mold_id[i]})'), row=3, col=1)
    
#     # Update subplot layout
#     fig.update_layout(title='Time Series Plots', showlegend=True)
    
#     # Save the plot
#     plot_file = os.path.join(output_folder, 'time_series_plots.html')
#     fig.write_html(plot_file)
def plots(df, output_folder='Plots'):
    mold_id = df['MOLD_ID'].unique
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    tav_array, shot_count_array, ct_array = features(df)

    # Create subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=['TAV', 'SHOT_COUNT', 'CT'])
    
    # Add traces to subplots
    for i in range(len(tav_array)):
        fig.add_trace(go.Scatter(x=df['HOUR'], y=tav_array[i], mode='lines', name=f'TAV ({mold_id[i]})'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['HOUR'], y=shot_count_array[i], mode='lines', name=f'SHOT_COUNT ({mold_id[i]})'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['HOUR'], y=ct_array[i], mode='lines', name=f'CT ({mold_id[i]})'), row=3, col=1)
    
    # Update subplot layout
    fig.update_layout(title='Time Series Plots', showlegend=True)
    
    # Save the plot
    plot_file = os.path.join(output_folder, 'time_series_plots.html')
    fig.write_html(plot_file)