import numpy as np
from sklearn.linear_model import Lasso, LinearRegression

class LargeWeightsRegressor:
    def __init__(self, sensor_index=0):
        self.coef_ = None
        self.sensors_used = None
        self.indices_used = None
        self.sensor_index = sensor_index
        self.filter_model = LinearRegression(positive=True, fit_intercept=False) #Lasso(positive=True, alpha=1e-8)
        self.fit_model = LinearRegression(positive=True, fit_intercept=False)

    def choose_top_weight_indices(self, weights):
        sorted_indices = np.flip(weights.argsort())
        total_used = 0
        threshold = 0.8
        num_connections = 10
        indices_used = []
        for index in sorted_indices[:num_connections]:
            total_used += weights[index]
            indices_used.append(index)
            if total_used > threshold:
                break
        return np.array(indices_used)

    def fit(self, X, y):
        self.filter_model.fit(X, y)
        weights = self.filter_model.coef_
        self.indices_used = self.choose_top_weight_indices(weights)
        self.sensors_used = np.array([i if i < self.sensor_index else i + 1 for i in self.indices_used])
        filtered_X = X[:, self.indices_used]
        self.fit_model.fit(filtered_X, y)
        self.coef_ = self.fit_model.coef_

    def predict(self, X):
        filtered_X = X[:, self.indices_used]
        return self.fit_model.predict(filtered_X)
    
    def set_sensor_index(self, sensor_index: int):
        self.sensor_index = sensor_index
        self.coef_ = None
        self.sensors_used = None
        self.indices_used = None


def main():
    from ui import show_weights
    from file_io import write_weights
    from preproc import preprocess

    safe_trace_file = "inputs/reversed.csv"
    # data = preprocess(safe_trace_file)[:, 27:]
    # np.savetxt("inputs/temperatures.csv", data, delimiter=",")
    data = np.genfromtxt("inputs/temperatures.csv", delimiter=",", dtype=float)
    np.set_printoptions(suppress=True)
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(data, test_size=0.2)
    def X_Y_split(data: np.ndarray, i: int):
        X = np.delete(data, i, axis=1)
        Y = data[:, i].astype(float)
        return X, Y
    for sensor_index in range(2):
        X_train, y_train = X_Y_split(train, sensor_index)
        X_test, y_test = X_Y_split(test, sensor_index)
        model = LargeWeightsRegressor(sensor_index=sensor_index)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        residuals = np.abs(predictions - y_test)
        print(residuals.mean())
        # write_weights(MODEL)
        # show_weights(sensor_index)


if __name__ == "__main__":
    main()