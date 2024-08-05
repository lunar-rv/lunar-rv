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
        threshold = 0.6
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

    # safe_trace_file = "csv/reversed.csv"
    # data = preprocess(safe_trace_file)[:, :27]
    data = np.genfromtxt("csv/pressures.csv", delimiter=",", dtype=float)
    np.set_printoptions(suppress=True)
    for sensor_index in range(2,5):
        print("Sensor index:", sensor_index+1)
        X = np.delete(data, sensor_index, axis=1)
        y = data[:, sensor_index].astype(float)
        MODEL = LargeWeightsRegressor(sensor_index=sensor_index)
        MODEL.fit(X, y)
        print(MODEL.coef_)
        print(MODEL.sensors_used)
        # write_weights(MODEL)
        # show_weights(sensor_index)


if __name__ == "__main__":
    main()