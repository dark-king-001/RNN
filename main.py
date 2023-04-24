# %% activating conda enviornment
import os
os.system("conda activate tf")
# %% Activating tensorflow
import tensorflow as tf 
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# %%
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %%
# Load data from CSV file
df = pd.read_csv('datasets/monthly_milk_production.csv')
# %%
# Reshape data for input to RNN
data = np.array(df['Production']).reshape(168,1, 1) 
print(data[0])
dates = np.array(range(0,len(data),1)).reshape(168,1, 1)
print(dates[0])

# %%
# # Create input sequences and targets
# input_seqs = np.zeros((num_samples, sequence_length, 1))
# targets = np.zeros(num_samples)
# for i in range(num_samples):
#     input_seqs[i] = data[i:i+sequence_length]
#     targets[i] = target[i]
# %%
# Create the RNN model
model = Sequential()
model.add(SimpleRNN(8, input_shape=(1, 1)))
model.add(SimpleRNN(8, input_shape=(1, 1)))
model.add(Dense(1, activation='sigmoid'))
# %%
# Compile the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# %%
# Train the model
history = model.fit(dates, data, epochs=500)
# %%
# Plot the training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.legend()
plt.show()
# %%
# Make predictions
y_pred = model.predict([[300]])

# Print the predicted values
print(y_pred)
# %%
