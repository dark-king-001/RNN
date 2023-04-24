# %% Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
import tensorflow as tf 
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# %% Load data from CSV file
df = pd.read_csv('datasets/monthly_milk_production.csv')
df.head()
# %% Convert date column to datetime object
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m')
df.head()
# %% Set date column as index
df.set_index('Date', inplace=True)
df.head()
# %% Resample the data at monthly frequency
df = df.resample('M').sum()
df.head()

# %% Split data into training and testing sets
train_size = int(len(df) * 0.8)
train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]

# %% Scale the data between 0 and 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)
print(train_data)
print(test_data)
# %% Define the function to create the input and output sequences for the RNN
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i])
    X, y = np.array(X), np.array(y)
    return X, y

# %% Define the sequence length
sequence_length = 3

# %% Create the training sequences
X_train, y_train = create_sequences(train_data, sequence_length)
print(X_train)
print(y_train)
# %% Create the RNN model
model = Sequential()
model.add(SimpleRNN(64, input_shape=(sequence_length, 1)))
model.add(Dense(1, activation='linear'))

# %% Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# %% Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=1)

# %% Plot the training history
plt.plot(history.history['loss'], label='Training Loss')
plt.legend()
plt.show()

# %% Create the testing sequences
X_test, y_test = create_sequences(test_data, sequence_length)

# %% Make predictions
y_pred = model.predict(X_test)

# %% Rescale the data back to its original form
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

# %% Plot the predictions against the actual values
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()
# %%
