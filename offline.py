import json
from synth import positive_synth
from evaluation import cut, X_Y_split
from regressor import LargeWeightsRegressor
from datetime import datetime, timedelta
import numpy as np
from preproc import preprocess
from ui import print_anomaly_info
with open("config.json") as config_file:
    config = json.load(config_file)

def offline_ad(infile="inputs/reversed.csv", sensor_type="PRESSURE"):
    with open(infile, "r") as file:
        raw_data = file.readlines()
    second_line = raw_data[1].split(";")
    date = second_line[1]
    time = second_line[2]
    batch_start_time = datetime.strptime(date + " " + time, "%d/%m/%Y %H:%M:%S")
    data = preprocess(infile)
    num_sensors_used = 27
    indices_used = np.arange(num_sensors_used) if sensor_type=="PRESSURE" else np.arange(num_sensors_used, 2*num_sensors_used)
    data_used = data[:, indices_used]
    train_length = config["WARMUP_TIME"] * (60 // config["TIME_PERIOD"]) * 24
    train_lr = data_used[:train_length, :]
    train_stl = data_used[train_length:train_length*2, :]
    batch_size = config["BATCH_SIZE"]
    for sensor_index in range(num_sensors_used):
        X_train_lr, Y_train_lr = X_Y_split(train_lr, sensor_index)
        X_train_stl, Y_train_stl = X_Y_split(train_stl, sensor_index)
        model = LargeWeightsRegressor()
        model.set_sensor_index(sensor_index)
        model.fit(X_train_lr, Y_train_lr)
        predictions = model.predict(X_train_stl)
        residuals = np.abs(predictions - Y_train_stl) * 1000
        residuals = cut(residuals, batch_size)
        residuals = residuals.reshape(-1, batch_size)
        formula = positive_synth(residuals)
        print(f"Sensor {sensor_index+1} formula: {formula}")
        test = data_used[train_length:, :]
        X_test, Y_test = X_Y_split(test, sensor_index)
        test_residuals = np.abs(model.predict(X_test) - Y_test) * 1000
        test_residuals = cut(test_residuals, batch_size)
        test_residuals = test_residuals.reshape(-1, batch_size)
        evaluation = formula.evaluate3(test_residuals, labels=False)
        rob = evaluation.mean(axis=1) if config["USE_MEAN"] else evaluation.min(axis=1)
        anomalies = np.where(rob < 0)[0]
        for i in anomalies:
            train_length *= num_sensors_used * 2
            anom_index = (int(i) * batch_size * num_sensors_used * 2 + train_length * 2)
            batch = raw_data[anom_index+1:anom_index+1+batch_size*num_sensors_used*2]
            anomaly_time = batch_start_time + timedelta(minutes=anom_index * config["TIME_PERIOD"])
            print(f"Anomaly detected at {anomaly_time}")
            print_anomaly_info(model, batch, formula, sensor_type)
    ### Assuming the first N values are safe

if __name__ == "__main__":
    offline_ad()