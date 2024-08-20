import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from residuals_over_time import X_Y_split, cut
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor
from scipy.stats import anderson, shapiro, norm, zscore
import statsmodels.api as sm


def get_residuals():
    pressures = np.genfromtxt("inputs/pressures.csv", delimiter=",")
    pressures = cut(pressures, 96)
    batch_pressures = pressures.reshape(-1, 27, 96)
    model = LinearRegression(fit_intercept=False, positive=True)
    all_residuals = []

    window_size = 30  # Define the window size for rolling predictions

    for i in range(27):  # Iterate over each sensor
        print("SENSOR", i)
        sensor_residuals = []

        for time_index in range(96):  # Iterate over each time of day (1 to 96)
            data = batch_pressures[:, i, time_index]  # Select all readings for the current time of day

            # Prepare training data using a modified rolling window approach
            for batch_num in range(len(data)):
                if batch_num < window_size:
                    # Use future values to fill the training window if batch_num < window_size
                    if batch_num + window_size >= len(data):
                        break
                    X_train = data[batch_num + 1:batch_num + 1 + window_size].reshape(-1, 1)
                    y_train = data[batch_num + 2:batch_num + 2 + window_size]  # Shift y_train by one time step
                else:
                    # Use past values for training once enough data is available
                    X_train = data[batch_num - window_size:batch_num].reshape(-1, 1)
                    y_train = data[batch_num - window_size + 1:batch_num + 1]  # Shift y_train by one time step

                X_test = np.array([[data[batch_num]]])  # Current day's reading
                y_test = data[batch_num + 1] if batch_num + 1 < len(data) else None  # Next day's reading

                if y_test is not None:
                    model.fit(X_train, y_train)
                    prediction = model.predict(X_test)[0]
                    residual = prediction - y_test
                    sensor_residuals.append(residual)

        all_residuals.append(sensor_residuals)
    np.save("sensor_wise_pressure.npy", np.array(all_residuals))

def sgd_residuals():
    pressures = np.genfromtxt("inputs/pressures.csv", delimiter=",")
    pressures = cut(pressures, 96)  # Assuming cut is a custom function
    batch_pressures = pressures.reshape(-1, 27, 96)
    all_residuals = []
    
    for i in range(27):
        print("SENSOR", i)
        X = np.delete(batch_pressures, i, axis=1)
        Y = batch_pressures[:, i, :].astype(float)
        sensor_residuals = []
        
        # Initialize the SGDRegressor outside the loop
        model = SGDRegressor(max_iter=1, tol=None, learning_rate='invscaling', eta0=0.01, power_t=0.25)
        
        for batch_num in range(batch_pressures.shape[0]):
            if batch_num == 0:
                continue
            
            X_train = X[batch_num - 1].reshape(-1, 26)  # Previous batch as input
            y_train = Y[batch_num - 1].reshape(-1)  # Previous batch as target
            
            # Update the model with the latest data
            model.partial_fit(X_train, y_train)
            
            # Predict on the current batch
            X_test = X[batch_num].reshape(-1, 26)  # Current batch as input
            y_test = Y[batch_num].reshape(-1)  # Current batch as target
            predictions = model.predict(X_test)
            residuals = np.abs(predictions - y_test)
            sensor_residuals.append(residuals.mean())
        
        all_residuals.append(sensor_residuals)
    
    np.save("sgd_pressure_residuals.npy", np.array(all_residuals))

def plot_residuals():
    # get_residuals()
    all_residuals = np.load("dt_pressure.npy")
    # sgd_residuals = np.load("sgd_pressure_residuals.npy")
    plt.title("Residuals of decision tree trained on all data (Pressure)")
    plt.xlabel("Day")
    plt.ylabel("Mean Absolute Residuals")
    # plt.plot(all_residuals.mean(axis=0), label="Using all values")
    # plt.plot(window_residuals.mean(axis=0), label="Using window")
    for i in range(27):
        plt.plot(all_residuals.T[:, i], label=f"Sensor {i+1}")
    # plt.legend()
    plt.show()
    # plt.savefig("temp_anomaly.png")

def hetero():
    pressures = cut(np.genfromtxt("inputs/pressures.csv", delimiter=","), 96)
    pressures = pressures.reshape(-1, 96, 27)  # Shape it into (days, time_of_day, sensors)
   
    days = pressures.shape[0]  # Number of days
    sensors = pressures.shape[2]  # Number of sensors

    # Array to hold the count of anomalies per day
    anomalies_per_day = np.zeros(days)

    for sensor_index in range(sensors):
        # Use other sensors to predict the current sensor's values
        for day in range(days):
            # Extract the day's data for all sensors
            daily_data = pressures[day, :, :]
            X = np.delete(daily_data, sensor_index, axis=1)  # All other sensors as features
            y = daily_data[:, sensor_index]  # The current sensor as the target
            
            # Fit a linear model using GLS to account for heteroscedasticity
            model = sm.GLS(y, sm.add_constant(X)).fit()
            y_pred = model.predict(sm.add_constant(X))
            
            # Calculate residuals
            residuals = y - y_pred
            
            # Estimate local variance (heteroscedasticity)
            local_variance = np.var(residuals)
            
            # Standardize residuals by dividing by local standard deviation
            standardized_residuals = residuals / np.sqrt(local_variance)
            
            # Detect anomalies: standardized residuals > 3 or < -3
            anomalies = np.where(np.abs(standardized_residuals) > 3)[0]
            
            # Count anomalies for the day
            anomalies_per_day[day] += len(anomalies)
    print(anomalies_per_day.tolist())
    # Plotting the number of anomalies per day
    plt.figure(figsize=(10, 6))
    plt.plot(anomalies_per_day, label='Number of Anomalies')
    plt.title('Number of Anomalies Detected per Day (Accounting for Heteroscedasticity)')
    plt.xlabel('Day')
    plt.ylabel('Number of Anomalies')
    plt.legend()
    plt.show()

def lr():
    pressures = cut(np.genfromtxt("inputs/pressures.csv", delimiter=","), 96)
    pressures = pressures.reshape(-1, 96, 27)  # Shape it into (days, time_of_day, sensors)
   
    days = pressures.shape[0]  # Number of days
    sensors = pressures.shape[2]  # Number of sensors

    # Array to hold the count of anomalies per day
    anomalies_per_day = np.zeros(days)
    for sensor_index in range(sensors):
        # Prepare data: other sensors as features, current sensor as the target
        X = np.delete(pressures, sensor_index, axis=2).reshape(-1, sensors - 1)  # Features
        y = pressures[:, :, sensor_index].flatten()  # Target

        # Fit a predictive model (Linear Regression here, but you could use others like Random Forest, etc.)
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Calculate residuals
        residuals = y - y_pred
        
        # Detect anomalies
        threshold = 3 * np.std(residuals)
        anomalies = np.where(np.abs(residuals) > threshold)[0]
        
        # Map anomalies to days
        days_with_anomalies = anomalies // 96

        for day in days_with_anomalies:
            anomalies_per_day[day] += 1

    # Plotting the number of anomalies per day
    plt.figure(figsize=(10, 6))
    plt.plot(anomalies_per_day, label='Number of Anomalies')
    plt.title('Number of Anomalies Detected per Day')
    plt.xlabel('Day')
    plt.ylabel('Number of Anomalies')
    plt.legend()
    plt.show()

def plot_weights(sensor_index=3):
    model = LinearRegression()
    pressures = np.genfromtxt("inputs/pressures.csv", delimiter=",")
    X = np.delete(pressures, sensor_index, axis=1)[20:50]
    Y = pressures[20:50, sensor_index]
    model.fit(X, Y)
    weights = model.coef_
    plt.bar(range(len(weights)), weights)
    plt.xlabel('Feature Index')
    plt.ylabel('Weight')
    plt.title('Linear Regression Weights')
    plt.show()

if __name__ == "__main__":
    get_residuals()
    # isolation_forest()
    # stl()
    # hetero()
    # plot_weights(sensor_index=3)
    # plot_weights(sensor_index=4)

