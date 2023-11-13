from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import TimeDistributed


# Function to create the model
def model(X_train,y_train,optimizer='adam'):
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[2]
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(n_timesteps,n_features)))
    model.add(TimeDistributed(Dense(n_outputs, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model