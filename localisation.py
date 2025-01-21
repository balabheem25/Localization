import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming you have a dataset with the following columns:
# [time_of_flight, timestamp_sent, timestamp_received, sender_id, receiver_id, x_position, y_position]
# x_position, y_position are the target labels (location of the X-sensor)

# Load your dataset
# Replace 'your_dataset.csv' with the actual dataset file path
data = np.loadtxt('synthetic_ultrasound_dataset.csv', delimiter=',', skiprows=1)

# Split the data into features (X) and target (y)
X = data[:, :-2]  # All columns except the last two (x_position, y_position)
y = data[:, -2:]  # The last two columns (x_position, y_position)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(2, activation='linear')  # 2 outputs for x and y coordinates
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
epochs = 100
batch_size = 32
history = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=1)

# Evaluate the model on the test data
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

# Save the model
model.save('sensor_localization_model.h5')

# Example prediction
# Replace this with new input data for prediction
new_data = np.array([[0.005, 1623958392, 1623958392.005, 1, 2]])  # Example input
new_data_scaled = scaler.transform(new_data)
predicted_position = model.predict(new_data_scaled)
print(f"Predicted Position: {predicted_position}")



