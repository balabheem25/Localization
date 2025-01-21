import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense, Concatenate, Embedding, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Huber
from scipy.stats import norm

# Neural Network Model
def build_neural_network_model():
    # Input for time of flight
    tof_input = Input(shape=(1,), name="time_of_flight_input")
    tof_dense = Dense(128, activation='relu')(tof_input)

    # Input for timestamps (sent and received)
    timestamps_input = Input(shape=(2,), name="timestamps_input")
    timestamps_dense = Dense(128, activation='relu')(timestamps_input)

    # Input for sensor IDs (sender and receiver)
    ids_input = Input(shape=(2,), name="sensor_ids_input")
    ids_dense = Dense(8, activation='relu')(ids_input)

    # Concatenate all layers
    merged = Concatenate()([tof_dense, timestamps_dense, ids_dense])

    # Fully connected hidden layers
    hidden = Dense(64, activation='relu')(merged)
    hidden = Dropout(0.2)(hidden)
    hidden = Dense(32, activation='relu')(hidden)

    # Output layer
    output = Dense(2, activation='linear', name="position_output")(hidden)

    # Build the model
    model = Model(inputs=[tof_input, timestamps_input, ids_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model

# Particle Filter
def particle_filter(initial_position, measurements, num_particles=500, noise_std=0.1, iterations=10, epsilon=1e-8):
    particles = np.random.normal(initial_position, scale=noise_std, size=(num_particles, 2))
    weights = np.ones(num_particles) / num_particles

    for _ in range(iterations):
        # Predict step: Add random motion
        particles += np.random.normal(0, noise_std, particles.shape)

        # Update step: Compute weights based on measurements
        distances = np.linalg.norm(particles - measurements, axis = 1)
        likelihoods = norm.pdf(distances, loc=0, scale=noise_std)
        weights *= likelihoods
        weights_sum = np.sum(weights)

        # Normalize weights, ensuring no division by zero
        if weights_sum == 0 or np.isnan(weights_sum):
            weights = np.ones(num_particles) / num_particles
        else:
            weights /= weights_sum
        
        # Resample step
        indices = np.random.choice(range(num_particles), size=num_particles, p=weights)
        particles = particles[indices]
        weights = weights[indices]

    # Estimate position
    estimated_position = np.average(particles, weights=weights, axis=0)
    return estimated_position

# Load Dataset
dataset = pd.read_csv("synthetic_ultrasound_dataset.csv")
X_tof = dataset["time_of_flight"].values.reshape(-1, 1)
X_timestamps = dataset[["timestamp_sent", "timestamp_received"]].values
X_ids = dataset[["sender_id", "receiver_id"]].values
y = dataset[["x_position", "y_position"]].values

# Split Data
X_tof_train, X_tof_test, X_timestamps_train, X_timestamps_test, \
X_ids_train, X_ids_test, y_train, y_test = train_test_split(
    X_tof, X_timestamps, X_ids, y, test_size=0.2, random_state=42
)

# Standardize Data
scaler_tof = StandardScaler()
scaler_timestamps = StandardScaler()

X_tof_train = scaler_tof.fit_transform(X_tof_train)
X_tof_test = scaler_tof.transform(X_tof_test)
X_timestamps_train = scaler_timestamps.fit_transform(X_timestamps_train)
X_timestamps_test = scaler_timestamps.transform(X_timestamps_test)

# Train Neural Network
nn_model = build_neural_network_model()
nn_model.fit([X_tof_train, X_timestamps_train, X_ids_train], y_train, epochs=100, batch_size=32, verbose=1)

# Neural Network Prediction
y_pred_nn = nn_model.predict([X_tof_test, X_timestamps_test, X_ids_test])


def predict_particles(particles, motion_model, control_inputs, noise_scale=1.0):
    """
    Predict the next state of particles based on the motion model and control inputs,
    adding adaptive noise.

    Parameters:
        particles (np.ndarray): Current particle states (N x D array, where N is the number of particles).
        motion_model (callable): A function that predicts particle states given inputs.
        control_inputs (np.ndarray): Control inputs affecting the particles.
        noise_scale (float): Base scale for noise.
    
    Returns:
        np.ndarray: Updated particle states.
    """
    # Predict particle states based on the motion model and control inputs
    predicted_particles = motion_model(particles, control_inputs)

    # Calculate spread (standard deviation) of the particles
    particle_std = np.std(predicted_particles, axis=0)

    # Add adaptive Gaussian noise based on particle spread
    adaptive_noise = np.random.normal(0, particle_std * noise_scale, predicted_particles.shape)
    predicted_particles += adaptive_noise

    return predicted_particles


def motion_model(particles, control_inputs):
    # For simplicity, assume no external motion (identity transformation)
    return particles

particles = predict_particles(particles, motion_model, control_inputs=None, noise_scale=0.5)



# Particle Filter Refinement
refined_positions = np.array([
    particle_filter(pred, true, num_particles=500, noise_std=0.1, iterations=10)
    for pred, true in zip(y_pred_nn, y_test)
])
model.compile(optimizer='adam', loss=Huber(delta=1.0), metrics=['mae'])
# Evaluate Results
mse_nn = np.mean(np.linalg.norm(y_test - y_pred_nn, axis=1) ** 2)
mse_pf = np.mean(np.linalg.norm(y_test - refined_positions, axis=1) ** 2)

print(f"Neural Network MSE: {mse_nn:.4f}")
print(f"Particle Filter Refined MSE: {mse_pf:.4f}")
