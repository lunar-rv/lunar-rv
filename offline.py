import json
from synth import positive_synth
from evaluation import cut, X_Y_split
from regressor import LargeWeightsRegressor
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import numpy as np
from preproc import preprocess_trace
from ui import print_anomaly_info
with open("config.json") as config_file:
    config = json.load(config_file)

def offline_ad(parser, infile="inputs/traces.csv", batch_size = 96):
    with open(infile, "r") as file:
        raw_data = file.readlines()
    first_line = raw_data[1].split(",")
    date = first_line[-2]
    time = first_line[-1].strip()
    batch_start_time = datetime.strptime(date + " " + time, "%d/%m/%Y %H:%M:%S")
    data = preprocess_trace(infile=infile)
    num_sensors_used = parser.sensors // len(parser.type)
    for i, sensor_type in enumerate(parser.type):
        indices_used = np.arange(num_sensors_used * i, num_sensors_used * (i+1))
        data_used = data[:, indices_used]
        w_1 = config["WARMUP_1_TIME"] * parser.batch
        w_2 = config["WARMUP_2_TIME"] * parser.batch
        train_lr = data_used[:w_1, :]
        train_stl = data_used[w_1:w_2, :]
        test = data_used[w_1 + w_2:, :]
        total_num_anomalies = 0
        num_total = 0
        for sensor_index in range(2): # (num_sensors_used):
            X_train_lr, Y_train_lr = X_Y_split(train_lr, sensor_index)
            X_train_stl, Y_train_stl = X_Y_split(train_stl, sensor_index)
            model = LargeWeightsRegressor()
            # model.set_sensor_index(sensor_index)
            model.fit(X_train_lr, Y_train_lr)
            predictions = model.predict(X_train_stl)
            residuals = np.abs(predictions - Y_train_stl)
            residuals = cut(residuals, batch_size)
            residuals = residuals.reshape(-1, batch_size)
            formula = positive_synth(residuals)
            print(f"Sensor {sensor_index+1} formula: {formula}")
            X_train = np.vstack((X_train_lr, X_train_stl))
            Y_train = np.hstack((Y_train_lr, Y_train_stl))
            model.fit(X_train, Y_train)
            X_test, Y_test = X_Y_split(test, sensor_index)
            X_test = cut(X_test, batch_size)
            Y_test = cut(Y_test, batch_size)
            X_test = X_test.reshape(-1, batch_size, num_sensors_used - 1)
            Y_test = Y_test.reshape(-1, batch_size)
            num_anomalies = 0
            for X, Y in zip(X_test, Y_test):
                Y += 0.000
                predictions = model.predict(X)
                batch_residuals = np.abs(predictions - Y)
                evaluation = formula.evaluate(batch_residuals.reshape(1, -1), labels=False)
                rob = evaluation.min()
                if rob < 0:
                    num_anomalies += 1
                else:
                    X_train = np.vstack((X_train, X))
                    Y_train = np.hstack((Y_train, Y))
                    model.fit(X_train, Y_train)
                    residuals = np.vstack((residuals, batch_residuals))
                    formula = positive_synth(residuals)
            print(f"{num_anomalies}/{len(Y_test)} anomalies detected")
            total_num_anomalies += num_anomalies
            num_total += len(Y_test)
            # for i in anomalies:
            #     print(train_length)
            #     real_train_length = train_length * num_sensors_used * 2
            #     anom_index = (int(i) * batch_size * num_sensors_used * 2 + real_train_length * 2)
            #     print(anom_index+1)
            #     print(anom_index+1+batch_size*num_sensors_used*2)
            #     batch = raw_data[anom_index+1:anom_index+1+batch_size*num_sensors_used*2]
            #     print(batch[0])
            #     anomaly_time = batch_start_time + timedelta(minutes=(i + train_length) * config["TIME_PERIOD"])
            #     print(f"Anomaly detected at {anomaly_time}")
            #     print_anomaly_info(model, batch, formula, sensor_type)
        ### Assuming the first N values are safe
        print(f"For type {sensor_type}:")
        print(f"{total_num_anomalies}/{num_total} anomalies detected")
        print(f"Anomaly detection rate: {total_num_anomalies/num_total}")

if __name__ == "__main__":
    from parser import Parser
    parser = Parser()
    parser.parse("spec.file")
    offline_ad(parser)