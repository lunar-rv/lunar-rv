import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt


def plot(arr, boundary: float = None, title=""):
    plt.figure(figsize=(10, 6))
    plt.plot(arr)
    plt.xlabel("Trace")
    plt.ylabel("Robustness")
    if boundary:
        plt.axhline(boundary, color="red", label=f"Mean prediction error <= {boundary}")
    plt.title(title)
    plt.legend()
    plt.show()


def load_data(filename):
    traces = np.genfromtxt(filename, delimiter=",", dtype=float)
    traces = traces.reshape(27, -1, 96)  # Assuming the data reshaping is correct
    return traces


def build_autoencoder(input_shape):
    input_layer = Input(shape=(input_shape,))
    encoded = Dense(32, activation="relu")(input_layer)
    encoded = Dense(16, activation="relu")(encoded)

    decoded = Dense(32, activation="relu")(encoded)
    decoded = Dense(input_shape, activation="sigmoid")(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=2.5e-6), loss="mean_squared_error")
    return autoencoder


def train_autoencoder(autoencoder, data, epochs=2000, batch_size=32):
    autoencoder.fit(
        data,
        data,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=0,
        verbose=True,
    )


def load_new_data(filename):
    new_data = np.genfromtxt(filename, delimiter=",", dtype=float)
    return new_data.reshape(27, -1, 96)[0]


def detect_anomalies(autoencoder, new_data, threshold=0.02):
    reconstructions = autoencoder.predict(new_data)
    reconstruction_errors = np.mean(np.square(new_data - reconstructions), axis=1)
    print("RE:", reconstruction_errors)
    anomalies = reconstruction_errors[reconstruction_errors > threshold]
    return anomalies, reconstruction_errors


def main():
    # # Load and prepare the data
    traces = load_data("csv/predictions.csv")
    normal_data = traces.reshape(-1, 96)  # Flatten the data for training

    # Build and train the autoencoder
    autoencoder = build_autoencoder(input_shape=96)
    train_autoencoder(autoencoder, normal_data)

    # Save the trained model
    autoencoder.save("autoencoder_model.h5")

    # Load the trained model
    autoencoder = tf.keras.models.load_model("autoencoder_model.h5")

    # Load new data for evaluation
    new_data = load_new_data("csv/negative_val.csv")
    anomalies, errors = detect_anomalies(autoencoder, new_data)

    # mean_errors = errors.mean(axis=1)
    plot(errors, boundary=0.02)
    neg_accuracy = (len(new_data) - anomalies.size) / len(new_data)
    print("Accuracy on negative traces:", neg_accuracy)
    # Load new data for evaluation
    new_data = load_new_data("csv/positive_val.csv")
    anomalies, errors = detect_anomalies(autoencoder, new_data)
    # mean_errors = errors.mean(axis=1)
    plot(errors, boundary=0.02)
    pos_accuracy = anomalies.size / len(new_data)
    print("Accuracy on positive traces:", pos_accuracy)
    print(f"Overall accuracy: {(pos_accuracy + neg_accuracy) / 2}")


if __name__ == "__main__":
    main()
