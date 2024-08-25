import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import IsolationForest
from residuals_over_time import X_Y_split, cut
from scipy.stats import zscore, shapiro, anderson
import scipy.stats as stats

def combined_anomaly_detection():
    # pressures = cut(np.genfromtxt("inputs/pressures.csv", delimiter=","), 96)
    pressures = np.load("sensor_wise_pressure.npy")
    pressures = pressures.reshape(-1, 96, 27)  # Shape it into (days, time_of_day, sensors)
   
    days = pressures.shape[0]  # Number of days
    sensors = pressures.shape[2]  # Number of sensors

    # Arrays to hold the count of anomalies per day for each method
    anomalies_per_day_stl = np.zeros(days)
    anomalies_per_day_if = np.zeros(days)
    anomalies_per_day_z = np.zeros(days)

    # STL Method
    for sensor_index in range(sensors):
        sensor_data = pressures[:, :, sensor_index].flatten()
        
        # Convert it into a pandas series for ease of use
        sensor_series = pd.Series(sensor_data)
        
        # Perform seasonal decomposition
        result = seasonal_decompose(sensor_series, period=96, model='additive', extrapolate_trend='freq')
        
        # Get the residuals
        residuals = result.resid
        
        # Define threshold for anomalies
        threshold = 3 * np.std(residuals)
        
        # Find anomalies
        anomalies = np.where(np.abs(residuals) > threshold)[0]
        
        # Map anomalies to days
        days_with_anomalies = anomalies // 96
        
        # Count anomalies per day
        for day in days_with_anomalies:
            anomalies_per_day_stl[day] += 1

    # Isolation Forest Method
    batch_pressures = pressures.reshape(-1, 27, 96)
    reshaped_pressures = batch_pressures.transpose(0, 2, 1).reshape(-1, 27)

    iso_forest = IsolationForest(contamination='auto', random_state=42)
    iso_forest.fit(reshaped_pressures)
    predictions = iso_forest.predict(reshaped_pressures)
    predictions_reshaped = predictions.reshape(batch_pressures.shape[0], 96)
    anomalies_per_day_if = np.sum(predictions_reshaped == -1, axis=1)

    # Z-Score Method
    for sensor_index in range(sensors):
        sensor_data = pressures[:, :, sensor_index].flatten()
        z_scores = zscore(sensor_data)
        anomalies = np.where(np.abs(z_scores) > 3)[0]
        days_with_anomalies = anomalies // 96
        for day in days_with_anomalies:
            anomalies_per_day_z[day] += 1

    # Plotting the number of anomalies per day for all methods
    plt.figure(figsize=(10, 6))
    plt.plot(anomalies_per_day_stl, label='STL Anomalies')
    plt.plot(anomalies_per_day_if, label='Isolation Forest Anomalies')
    plt.plot(anomalies_per_day_z, label='Z-Score Anomalies')
    plt.title('Number of Anomalies Detected per Day (Temperature)')
    plt.xlabel('Day')
    plt.ylabel('Number of Anomalies')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    combined_anomaly_detection()