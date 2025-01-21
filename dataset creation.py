import numpy as np
import pandas as pd

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

for _ in range(num_samples):
    # Randomly select the X-sensor position within the circle
    r = radius * np.sqrt(np.random.rand())
    theta = np.random.rand() * 2 * np.pi
    x_position = r * np.cos(theta)
    y_position = r * np.sin(theta)

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
dataset.to_csv("/Users/balabheem/Downloads/VScode files/synthetic_ultrasound_dataset.csv", index=False)
print("Dataset generated and saved to 'synthetic_ultrasound_dataset.csv'")
