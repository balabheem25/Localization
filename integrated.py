import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Parameters for the setup
num_sensors = 8
radius = 10  # Radius in cm (diameter = 20 cm)
frequency = 40e3  # Ultrasound frequency in Hz (example: 40 kHz)
signal_speed = 34300  # Speed of sound in cm/s
num_samples = 1000  # Number of samples to generate

# Generate sensor positions in a circular 2D space
def generate_sensor_positions(num_sensors, radius):
    angles = np.linspace(0, 2 * np.pi, num_sensors, endpoint=False)
    positions = [(radius * np.cos(angle), radius * np.sin(angle)) for angle in angles]
    return np.array(positions)

sensor_positions = generate_sensor_positions(num_sensors, radius)

# Generate dataset
np.random.seed(42)  # For reproducibility
samples = []
x_positions = []
y_positions = []

for _ in range(num_samples):
    # Randomly select the X-sensor position within the circle
    r = radius * np.sqrt(np.random.rand())
    theta = np.random.rand() * 2 * np.pi
    x_position = r * np.cos(theta)
    y_position = r * np.sin(theta)
    x_positions.append(x_position)
    y_positions.append(y_position)

    # Compute time of flight for each sensor
    times_of_flight = []
    for sensor_x, sensor_y in sensor_positions:
        distance = np.sqrt((sensor_x - x_position) ** 2 + (sensor_y - y_position) ** 2)
        time_of_flight = distance / signal_speed
        times_of_flight.append(time_of_flight)

    # Synchronize timestamps (all sensors have the same initial timestamp)
    timestamp_sent = np.random.uniform(0, 1)  # Random timestamp between 0 and 1 second
    timestamps_received = [timestamp_sent + tof for tof in times_of_flight]

    # Generate records for all sensor pairs
    for sender_id in range(num_sensors):
        for receiver_id in range(num_sensors):
            if sender_id != receiver_id:
                samples.append([
                    times_of_flight[sender_id],
                    timestamp_sent,
                    timestamps_received[receiver_id],
                    sender_id,
                    receiver_id,
                    x_position,
                    y_position
                ])

# Convert samples to a DataFrame
dataset = pd.DataFrame(samples, columns=[
    "time_of_flight", "timestamp_sent", "timestamp_received", 
    "sender_id", "receiver_id", "x_position", "y_position"
])

# Save to a CSV file
dataset.to_csv("synthetic_ultrasound_dataset.csv", index=False)
print("Dataset generated and saved to 'synthetic_ultrasound_dataset.csv'")

# Visualization of sensor setup
plt.figure(figsize=(8, 8))
plt.scatter(sensor_positions[:, 0], sensor_positions[:, 1], color='blue', label='Sensors')
plt.gca().add_artist(plt.Circle((0, 0), radius, fill=False, linestyle='--', color='gray'))
plt.title("Sensor Setup in 2D Circular Space")
plt.xlabel("X (cm)")
plt.ylabel("Y (cm)")
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.legend()
plt.grid(True)
plt.show()

# Animation of X-sensor positions
def animate(i):
    plt.clf()
    plt.scatter(sensor_positions[:, 0], sensor_positions[:, 1], color='blue', label='Sensors')
    plt.scatter(x_positions[i], y_positions[i], color='red', label='X-Sensor')
    plt.gca().add_artist(plt.Circle((0, 0), radius, fill=False, linestyle='--', color='gray'))
    plt.title("Sensor Interaction Animation")
    plt.xlabel("X (cm)")
    plt.ylabel("Y (cm)")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.grid(True)

ani = FuncAnimation(plt.gcf(), animate, frames=min(len(x_positions), 100), interval=500)
plt.show()

# Load dataset for localization model
X = dataset.iloc[:, :-2].values  # All columns except the last two (x_position, y_position)
y = dataset.iloc[:, -2:].values  # The last two columns (x_position, y_position)

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
