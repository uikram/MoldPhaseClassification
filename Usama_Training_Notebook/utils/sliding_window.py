import numpy as np
import pandas as pd


#Create a sliding window
def sliding_window(df, window_size):
    df['HOUR'] = pd.to_datetime(df['HOUR'], format='%Y-%m-%d %H:%M:%S')
    df = df.sort_values('HOUR')
    # One-hot encoding the 'PHASE' feature
    y = pd.get_dummies(df['PHASE']).values
    X=df[['CT', 'TAV', 'SHOT_COUNT']]
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        v = X.iloc[i:(i + window_size)].values
        Xs.append(v)
        ys.append(y[i:i + window_size])
    return np.array(Xs), np.array(ys)