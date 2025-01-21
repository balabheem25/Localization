import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Input, Dense, Concatenate, Embedding, Flatten, Dropout
from tensorflow.keras.models import Model, load_model
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Redefine the neural network model with specialized layers
def build_specialized_model():
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

dataset = pd.read_csv("/Users/balabheem/Downloads/VScode files/synthetic_ultrasound_dataset.csv")
# Prepare inputs for the specialized model
X_tof = dataset["time_of_flight"].values.reshape(-1, 1)
X_timestamps = dataset[["timestamp_sent", "timestamp_received"]].values
X_ids = dataset[["sender_id", "receiver_id"]].values
y = dataset[["x_position", "y_position"]].values

# Split the dataset
X_tof_train, X_tof_test, X_timestamps_train, X_timestamps_test, \
X_ids_train, X_ids_test, y_train, y_test = train_test_split(
    X_tof, X_timestamps, X_ids, y, test_size=0.2, random_state=42
)

# Standardize numerical inputs
scaler_tof = StandardScaler()
scaler_timestamps = StandardScaler()

X_tof_train = scaler_tof.fit_transform(X_tof_train)
X_tof_test = scaler_tof.transform(X_tof_test)

X_timestamps_train = scaler_timestamps.fit_transform(X_timestamps_train)
X_timestamps_test = scaler_timestamps.transform(X_timestamps_test)


# Build and train the specialized model
specialized_model = build_specialized_model()
history = specialized_model.fit(
    [X_tof_train, X_timestamps_train, X_ids_train], y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    verbose=1
)

# Evaluate the model
test_loss, test_mae = specialized_model.evaluate(
    [X_tof_test, X_timestamps_test, X_ids_test], y_test, verbose=0
)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

# Save the specialized model
specialized_model.save('specialized_sensor_localization_model.h5')

# Example prediction
new_data_tof = np.array([[0.005]])  # Example TOF
new_data_timestamps = np.array([[1623958392, 1623958392.005]])  # Example timestamps
new_data_ids = np.array([[1, 2]])  # Example sender and receiver IDs

new_data_tof_scaled = scaler_tof.transform(new_data_tof)
new_data_timestamps_scaled = scaler_timestamps.transform(new_data_timestamps)

predicted_position = specialized_model.predict(
    [new_data_tof_scaled, new_data_timestamps_scaled, new_data_ids]
)
print(f"Predicted Position: {predicted_position}")

#########################################################################
#3 figs, 10 points fro each image
number_of_samples_per_figure = 10
number_of_figures = 3
indices = np.random.choice(len(y_test), number_of_samples_per_figure*number_of_figures, replace=False)
selected_indices = (indices, number_of_figures)



#########################################################################
# Evaluation Metrics and Visualization
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    accuracy = np.mean(np.linalg.norm(y_true - y_pred, axis=1) < 1)  # Accuracy within 1 cm
    return mse, accuracy

# Calculate metrics
#mse, accuracy = calculate_metrics(y_test, y_pred)

def plot_actual_vs_predicted(y_true, y_pred, figure_number, mse, accuracy):
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true[:, 0], y_true[:, 1], color='blue', label='Actual Positions')
    plt.scatter(y_pred[:, 0], y_pred[:, 1], color='red', label='Predicted Positions', alpha=0.7)
    plt.title(f"Actual vs Predicted X-Sensor Positions (Figure {figure_number}) \n"
              f"MSE: {mse:.4f}, Accuracy: {accuracy * 100:.2f}%")
    plt.xlabel("X (cm)")
    plt.ylabel("Y (cm)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Predictions on test data. jkdsfgnkdnvgfkf code commented
#y_pred = specialized_model.predict([X_tof_test,X_timestamps_test,X_ids_test])

# Calculate metrics 1323778327374878code commented
#mse, accuracy = calculate_metrics(y_test, y_pred)
#print(f"Mean Squared Error (MSE): {mse}")
#print(f"Accuracy (within 1 cm): {accuracy * 100:.2f}%")


#Visualisation in different figures
for figure_number, idx in enumerate(selected_indices, start=1):
    y_true_subset  = y_test[idx]
    X_tof_subset = X_tof_test[idx]
    X_timestamps_subset = X_timestamps_test[idx]
    X_ids_subset = X_ids_test[idx]

# Predict for the subset
    y_pred_subset = specialized_model.predict([X_tof_subset, X_timestamps_subset, X_ids_subset])

    # Calculate metrics
    mse, accuracy = calculate_metrics(y_true_subset, y_pred_subset)

    # Plot actual vs predicted
    plot_actual_vs_predicted(y_true_subset, y_pred_subset, figure_number, mse, accuracy)


# Plot actual vs predicted positions akfbgkdsbfkdfkdjbf code edited
#plot_actual_vs_predicted(y_test, y_pred)



# Confusion Matrix (for a classification-like task, threshold-based)
threshold = 1  # Threshold for "correct" prediction
#distances = np.linalg.norm(y_test - y_pred, axis=1) hjbvdfmnbdfmbdf changed
distances = np.linalg.norm(y_test - specialized_model([X_tof_test, X_timestamps_test, X_ids_test]), axis=1)
confusion_labels = distances < threshold
confusion_mat = confusion_matrix([1] * len(confusion_labels), confusion_labels.astype(int))
print("Confusion Matrix:")
print(confusion_mat)

# Update Animation with Predictions
def animate_with_predictions(i):
    plt.clf()
    plt.scatter(sensor_positions[:, 0], sensor_positions[:, 1], color='blue', label='Sensors')
    plt.scatter(x_positions[i], y_positions[i], color='green', label='Actual X-Sensor Position')
    plt.scatter(y_pred[i, 0], y_pred[i, 1], color='red', label='Predicted Position', alpha=0.7)
    plt.gca().add_artist(plt.Circle((0, 0), radius, fill=False, linestyle='--', color='gray'))
    plt.title("Sensor Interaction Animation with Predictions")
    plt.xlabel("X (cm)")
    plt.ylabel("Y (cm)")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.grid(True)

ani = FuncAnimation(plt.gcf(), animate_with_predictions, frames=min(len(x_positions), 100), interval=500)
plt.show()