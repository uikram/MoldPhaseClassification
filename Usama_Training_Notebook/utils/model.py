import os
import datetime
import wandb
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from keras import metrics
from wandb.keras import WandbCallback
from keras.callbacks import ModelCheckpoint,Callback
import numpy as np
import glob


os.environ["WANDB_NOTEBOOK_NAME"] = "Training.ipynb"

# # Function to create the model
# def model(X_train,y_train,optimizer):
#     # Initialize wandb
#     run = wandb.init(project='MPD', entity='DeepChain')

#     n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[2]
#     model = Sequential()
#     model.add(LSTM(50, return_sequences=True, input_shape=(n_timesteps,n_features)))
#     model.add(TimeDistributed(Dense(n_outputs, activation='softmax')))
#     model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', metrics.Precision(), metrics.Recall()])

#     # Add early stopping
#     es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

#     # Add ModelCheckpoint
#     mc = ModelCheckpoint('Weights/best_model.h5', monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True)

#     # Add TensorBoard logs
#     log_dir = os.path.join(
#         "logs",
#         datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
#     )
#     tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

#     # Add wandb callback
#     wandb_callback = WandbCallback()
    
#     # If exists a best model, load its weights!
#     if os.path.isfile('best_model.h5'):
#         model.load_weights('best_model.h5')

#     # Add the callbacks to the model
#     model.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[es, mc, tensorboard_callback, wandb_callback])
    
#     return model



# import os
# import glob
# from keras.callbacks import ModelCheckpoint, Callback

# os.environ["WANDB_NOTEBOOK_NAME"] = "Training.ipynb"

# Custom callback for early stopping when both accuracy and loss are constant
class EarlyStoppingBoth(Callback):
    def __init__(self, patience=0):
        super(EarlyStoppingBoth, self).__init__()
        self.patience = patience
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
        self.best_loss = np.Inf
        self.best_acc = 0

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        current_acc = logs.get('val_accuracy')
        if np.less(current_loss, self.best_loss) and np.greater(current_acc, self.best_acc):
            self.best_loss = current_loss
            self.best_acc = current_acc
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best_loss = np.Inf
        self.best_acc = 0
        self.best_weights = self.model.get_weights()

# Function to create the model
def model(X_train, y_train, optimizer):
    # Initialize wandb
    run = wandb.init(project='MPD', entity='DeepChain')
    
    n_timesteps, n_features = X_train.shape[1], X_train.shape[2] 
    
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(n_timesteps, n_features)))  # Changed from return_sequences=False
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))  # Changed from Dense(1, activation='sigmoid')
    
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', metrics.Precision(), metrics.Recall()])  # No change in loss function

    # Add early stopping for both loss and accuracy
    es_both = EarlyStoppingBoth(patience=50)

    # Add ModelCheckpoint
    mc = ModelCheckpoint('Models/Epoch_{epoch:02d}.h5')

    # Add TensorBoard logs
    log_dir = os.path.join(
        "logs",
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Add wandb callback
    wandb_callback = WandbCallback()
    
    # If exists a best model, load its weights!
    list_of_files = glob.glob('Models/Epoch_*.h5')
    if list_of_files:  
        latest_file = max(list_of_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
        if os.path.isfile(latest_file):
            model.load_weights(latest_file)

    # Add the callbacks to the model
    model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[es_both, mc, tensorboard_callback, wandb_callback])
    
    return model

