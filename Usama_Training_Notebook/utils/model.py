from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import os
import datetime

# Function to create the model
def model(X_train,y_train,optimizer):
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[2]
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(n_timesteps,n_features)))
    model.add(TimeDistributed(Dense(n_outputs, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Add early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

    # Add TensorBoard logs
    log_dir = os.path.join(
        "logs",
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Add the callbacks to the model
    model.fit(X_train, y_train, epochs=30, validation_split=0.2, callbacks=[es, tensorboard_callback])

    return model