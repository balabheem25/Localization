import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Parameters
np.random.seed(42)
NUM_SENSORS = 8
NUM_PARTICLES = 1000
EPOCHS = 100
BATCH_SIZE = 32
RADIUS = 10  # Circular area radius



# Generate sensor positions (circle layout)
sensor_positions = np.array([
    [RADIUS * np.cos(2 * np.pi * i / NUM_SENSORS), RADIUS * np.sin(2 * np.pi * i / NUM_SENSORS)]
    for i in range(NUM_SENSORS)
])

# Generate synthetic dataset
X_data = []
y_data = []
for _ in range(1000):
    x_sensor = np.random.uniform(-RADIUS, RADIUS, size=2)
    distances = np.linalg.norm(sensor_positions - x_sensor, axis=1)
    X_data.append(distances)
    y_data.append(x_sensor)
X_data = np.array(X_data)
y_data = np.array(y_data)

# Split dataset into training and testing
split_idx = int(0.8 * len(X_data))
X_train, X_test = X_data[:split_idx], X_data[split_idx:]
y_train, y_test = y_data[:split_idx], y_data[split_idx:]

# Build neural network model
def build_model():
    model = Sequential([
        Input(shape=(NUM_SENSORS,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(2, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_model()

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

# Particle Filter Implementation
class ParticleFilter:
    def __init__(self, num_particles, sensor_positions, radius):
        self.num_particles = num_particles
        self.sensor_positions = sensor_positions
        self.radius = radius
        self.particles = self.initialize_particles()
        self.weights = np.ones(num_particles) / num_particles

    def initialize_particles(self):
        return np.random.uniform(-self.radius, self.radius, size=(self.num_particles, 2))


    def predict(self, noise_std=0.5):
        self.particles += np.random.normal(0, noise_std, self.particles.shape)

    def update(self, measurement, measurement_std):
        distances = np.linalg.norm(self.particles[:, np.newaxis, :] - self.sensor_positions, axis=2)
        predicted_distances = np.linalg.norm(self.particles[:, np.newaxis, :] - self.sensor_positions, axis=2)
        self.weights *= np.prod(np.exp(-0.5 * ((distances - measurement) / measurement_std) ** 2), axis=1)
        self.weights += 1e-300  # Avoid divide by zero
        self.weights /= np.sum(self.weights)

    def resample(self):
        indices = np.random.choice(np.arange(self.num_particles), size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate(self):
        return np.average(self.particles, weights=self.weights, axis=0)

# Particle filter initialization
pf = ParticleFilter(NUM_PARTICLES, sensor_positions, RADIUS)

# Visualization and Evaluation
def plot_results(actual, predicted_nn, predicted_pf, figure_number):
    plt.figure(figsize=(8, 8))
    plt.scatter(sensor_positions[:, 0], sensor_positions[:, 1], color='blue', label='Sensors')
    plt.scatter(actual[0], actual[1], color='green', label='Actual X-Sensor Position', s=100)
    plt.scatter(predicted_nn[0], predicted_nn[1], color='red', label='NN Predicted Position', s=100, alpha=0.7)
    plt.scatter(predicted_pf[0], predicted_pf[1], color='orange', label='PF Estimated Position', s=100, alpha=0.7)
    plt.gca().add_artist(plt.Circle((0, 0), RADIUS, fill=False, linestyle='--', color='gray'))
    plt.title(f"Localization Results (Figure {figure_number})")
    plt.xlabel("X (cm)")
    plt.ylabel("Y (cm)")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.grid(True)
    plt.show()

# Test and evaluate
predicted_nn_positions = model.predict(X_test)
confusion_labels = []
predicted_pf_positions = []

for i, (distances, actual_position) in enumerate(zip(X_test, y_test)):
    pf.particles = pf.initialize_particles()
    pf.weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES

    for _ in range(5):
        pf.predict()
        pf.update(distances, measurement_std=0.5)
        pf.resample()

    pf_position = pf.estimate()
    predicted_pf_positions.append(pf_position)

    # Threshold-based evaluation for confusion matrix
    nn_error = np.linalg.norm(actual_position - predicted_nn_positions[i])
    pf_error = np.linalg.norm(actual_position - pf_position)
    confusion_labels.append((nn_error < 1, pf_error < 1))

    if i < 3:  # Visualize the first 3 samples
        plot_results(actual_position, predicted_nn_positions[i], pf_position, i + 1)

predicted_pf_positions = np.array(predicted_pf_positions)

# Metrics and Confusion Matrix
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    accuracy = np.mean(np.linalg.norm(y_true - y_pred, axis=1) < 1)  # Accuracy within 1 cm
    return mse, accuracy

nn_mse, nn_accuracy = calculate_metrics(y_test, predicted_nn_positions)
pf_mse, pf_accuracy = calculate_metrics(y_test, predicted_pf_positions)

print(f"Neural Network - MSE: {nn_mse:.4f}, Accuracy: {nn_accuracy * 100:.2f}%")
print(f"Particle Filter - MSE: {pf_mse:.4f}, Accuracy: {pf_accuracy * 100:.2f}%")

confusion_nn = [label[0] for label in confusion_labels]
confusion_pf = [label[1] for label in confusion_labels]
confusion_mat = confusion_matrix(confusion_nn, confusion_pf)
print("Confusion Matrix:")
print(confusion_mat)

number_of_samples_per_figure = 10
number_of_figures = 3
indices = np.random.choice(len(y_test), number_of_samples_per_figure * number_of_figures, replace=False)
selected_indices = np.array_split(indices, number_of_figures)

#Predictions
#y_pred = model.predict([X_tof_test, X_timestamps_test, X_ids_test])

def plot_test_cases_subplots(y_test, predicted_positions, selected_indices, metrics, figure_title="Test Cases Subplots"):
    fig, axs = plt.subplots(1, len(selected_indices), figsize=(15, 5))
    
    for i, idx in enumerate(selected_indices):
        y_true_subset = y_test[idx]
        y_pred_subset = predicted_positions[idx]
        
        # Calculate metrics
        mse, accuracy = calculate_metrics(y_true_subset, y_pred_subset)
        metrics.append((mse, accuracy))
        
        # Plot actual vs predicted positions
        axs[i].scatter(y_true_subset[:, 0], y_true_subset[:, 1], color='blue', label='Actual Positions')
        axs[i].scatter(y_pred_subset[:, 0], y_pred_subset[:, 1], color='red', label='Predicted Positions', alpha=0.7)
        axs[i].set_title(f"Test Case {i + 1} \nMSE: {mse:.4f}, Accuracy: {accuracy * 100:.2f}%")
        axs[i].set_xlabel("X (cm)")
        axs[i].set_ylabel("Y (cm)")
        axs[i].legend()
        axs[i].grid(True)
    
    plt.suptitle(figure_title)
    plt.tight_layout()
    plt.show()

# Predictions
#y_pred = model.predict([X_tof_test, X_timestamps_test, X_ids_test])

# Plot subplots for test cases
metrics = []  # To store MSE and accuracy for each test case
plot_test_cases_subplots(y_test, predicted_nn_positions, selected_indices, metrics)

# Print metrics for each test case
for i, (mse, accuracy) in enumerate(metrics, start=1):
    print(f"Test Case {i}: MSE = {mse:.4f}, Accuracy = {accuracy * 100:.2f}%")
