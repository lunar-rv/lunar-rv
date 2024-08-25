
import json
from synth import positive_synth
from evaluation import cut, X_Y_split
from regressor import LargeWeightsRegressor
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import numpy as np
from preproc import preprocess_trace
from model import new_batch_ok, apply_anomaly
import matplotlib.pyplot as plt

with open("config.json") as config_file:
    config = json.load(config_file)

def plot_residuals(actual, predictions, sensor_index, sensor_type):
    plt.title(f"Predictions and actual values over time for {sensor_type.capitalize()} Sensor {sensor_index+1}")
    plt.plot(actual, label="Actual")
    plt.plot(predictions, label="Predictions")
    plt.legend()
    plt.show()

def offline_ad(parser, add_anomalies=False):
    infile = parser.infile
    batch_size = parser.batch
    with open(infile, "r") as file:
        raw_data = file.readlines()
    first_line = raw_data[1].split(",")
    second_line = raw_data[2].split(",")
    date_1 = first_line[-2]
    time_1 = first_line[-1].strip()
    time_2 = second_line[-1].strip()
    diff = datetime.strptime(time_1, "%H:%M:%S") - datetime.strptime(time_2, "%H:%M:%S")
    diff_minutes = diff.total_seconds() // 60
    batch_start_time = datetime.strptime(date_1 + " " + time_1, "%d/%m/%Y %H:%M:%S")
    data = preprocess_trace(infile=infile)
    warmup_1_time = int(config["WARMUP_1_PROPORTION"] * parser.safe)
    warmup_2_time = parser.safe - warmup_1_time
    for i, sensor_type in enumerate(parser.type):
        if i == 0:
            continue
        indices_used = np.arange(parser.type_indices[i], parser.type_indices[i+1])
        data_used = data[:, indices_used]
        w_1 = warmup_1_time * batch_size
        w_2 = warmup_2_time * batch_size
        train_lr = data_used[:w_1, :]
        train_stl = data_used[w_1:w_1+w_2, :]
        test = data_used[w_1 + w_2:, :]
        total_num_anomalies = 0
        num_total = 0
        num_sensors_used = len(indices_used)
        for sensor_index in range(num_sensors_used):
            X_train_lr, Y_train_lr = X_Y_split(train_lr, sensor_index)
            X_train_stl, Y_train_stl = X_Y_split(train_stl, sensor_index)
            # model.set_sensor_index(sensor_index)
            model = LargeWeightsRegressor()
            model.fit(X_train_lr, Y_train_lr)
            predictions = model.predict(X_train_stl)
            residuals = np.abs(predictions - Y_train_stl)
            if config["PLOT_RESIDUALS_GRAPHS"]:
                plot_residuals(actual=Y_train_stl, predictions=predictions, sensor_type=sensor_type, sensor_index=sensor_index)
            residuals = cut(residuals, batch_size)
            residuals = residuals.reshape(-1, batch_size)
            formula = positive_synth(residuals, operators=parser.stl)
            print(f"Sensor {sensor_index+1} formula: {formula}")
            X_train = np.vstack((X_train_lr, X_train_stl))
            Y_train = np.hstack((Y_train_lr, Y_train_stl))
            model.fit(X_train, Y_train)
            X_test, Y_test = X_Y_split(test, sensor_index)
            X_test = cut(X_test, batch_size)
            Y_test = cut(Y_test, batch_size)
            X_test = X_test.reshape(-1, batch_size, num_sensors_used - 1)
            Y_test = Y_test.reshape(-1, batch_size)
            if config["ADD_ANOMALIES_OFFLINE"]:
                anomaly_size = Y_test.std()
                Y_test += anomaly_size
            num_anomalies = 0
            current_time = batch_start_time + timedelta(parser.safe * diff_minutes * batch_size)
            index = w_1 + w_2
            for X, Y in zip(X_test, Y_test):
                new_batch = raw_data[index:index+batch_size]
                batch_residuals = np.abs(model.predict(X) - Y)
                if not new_batch_ok(batch_residuals, formula, new_batch=new_batch, sensor_index=sensor_index, sensor_type=sensor_type, print_info=False):
                    num_anomalies += 1
                    if not add_anomalies:
                        formula = positive_synth(residuals, operators=parser.stl)
                else:
                    residuals = np.vstack((residuals, batch_residuals))
                    X_train = np.vstack((X_train, X))
                    Y_train = np.hstack((Y_train, Y))
                    model.fit(X_train, Y_train) # refit model
                current_time += timedelta(diff_minutes * batch_size)
                index += batch_size
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
        print("=" * 65)
        print(f"For type {sensor_type}:")
        print(f"{total_num_anomalies}/{num_total} anomalies detected")
        print(f"Anomaly detection rate: {total_num_anomalies/num_total}")
        print("=" * 65)

if __name__ == "__main__":
    from parser import Parser
    parser = Parser()
    parser.parse("spec.file")
    offline_ad(parser)