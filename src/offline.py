
import json
from synth import positive_synth
from evaluation import cut, X_Y_split
from regressor import LargeWeightsRegressor
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import numpy as np
from preproc import preprocess_trace
from model import new_batch_ok#, apply_anomaly
import matplotlib.pyplot as plt
import pickle

with open("config.json") as config_file:
    config = json.load(config_file)

def plot_residuals(actual, predictions, sensor_index, sensor_type):
    plt.title(f"Predictions and actual values over time for {sensor_type.capitalize()} Sensor {sensor_index+1}")
    plt.plot(actual, label="Actual")
    plt.plot(predictions, label="Predictions")
    plt.legend()
    plt.show()

def offline_ad(parser):
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
        type_anomaly_times = []
        for sensor_index in range(num_sensors_used):
            anomaly_times = []
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
            X_test, Y_test = X_Y_split(test, sensor_index)
            X_test = cut(X_test, batch_size)
            Y_test = cut(Y_test, batch_size)
            X_test = X_test.reshape(-1, batch_size, num_sensors_used - 1)
            Y_test = Y_test.reshape(-1, batch_size)
            if config["ADD_ANOMALIES_OFFLINE"]:
                anomaly_size = 0#Y_test.std()
                Y_test += anomaly_size
            num_anomalies = 0
            current_time = batch_start_time + timedelta(parser.safe * diff_minutes * batch_size)
            index = w_1 + w_2
            for X, Y in zip(X_test, Y_test):
                new_batch = raw_data[index:index+batch_size]
                start = max(0, len(Y_train) - 30 * batch_size)
                X_train_window = X_train[start:]
                Y_train_window = Y_train[start:]
                model.fit(X_train_window, Y_train_window) # refit model
                batch_residuals = np.abs(model.predict(X) - Y)
                if not new_batch_ok(batch_residuals, formula, new_batch=new_batch, sensor_index=sensor_index, sensor_type=sensor_type, print_info=False):
                    num_anomalies += 1
                    if not config["ADD_ANOMALIES_OFFLINE"]:
                        formula = positive_synth(residuals, operators=parser.stl)
                    anomaly_times.append(index // batch_size)
                else:
                    residuals = np.vstack((residuals, batch_residuals))
                    X_train = np.vstack((X_train, X))
                    Y_train = np.hstack((Y_train, Y))
                current_time += timedelta(diff_minutes * batch_size)
                index += batch_size
            print(f"{num_anomalies}/{len(Y_test)} anomalies detected")
            total_num_anomalies += num_anomalies
            num_total += len(Y_test)
            type_anomaly_times.append(anomaly_times)
        
        with open(f"anomaly_times_{sensor_type}.pkl", "wb") as f:
            pickle.dump(type_anomaly_times, f)
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



def testing_1(parser, add_anomalies=False):
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
    proportion_data = np.empty((2, 9))
    for warmup_proportion in range(2, 20, 2):
        warmup_1_time = warmup_proportion
        warmup_2_time = 20 - warmup_1_time
        for i, sensor_type in enumerate(parser.type):
            indices_used = np.arange(parser.type_indices[i], parser.type_indices[i+1])
            data_used = data[:, indices_used]
            w_1 = warmup_1_time * batch_size
            w_2 = warmup_2_time * batch_size
            train_lr = data_used[:w_1, :]
            train_stl = data_used[w_1:w_1+w_2, :]
            train = data_used[:w_1+w_2]
            test = data_used[w_1+w_2:, :]
            total_num_anomalies = 0
            num_total = 0
            num_sensors_used = len(indices_used)
            all_residuals = []
            for sensor_index in range(num_sensors_used):
                X_train_lr, Y_train_lr = X_Y_split(train_lr, sensor_index)
                X_train_stl, Y_train_stl = X_Y_split(train_stl, sensor_index)
                # model.set_sensor_index(sensor_index)
                model = LinearRegression()
                model.fit(X_train_lr, Y_train_lr)
                predictions = model.predict(X_train_stl)
                residuals = np.abs(predictions - Y_train_stl)
                all_residuals.append(residuals.mean())
                continue
            x = i
            y = int(warmup_proportion // 2) - 1
            proportion_data[x, y] = np.mean(all_residuals)
    for data in proportion_data:
        print(data.tolist())
        
def testing_2(parser, add_anomalies=False):
    infile = parser.infile
    batch_size = parser.batch
    with open(infile, "r") as file:
        data = preprocess_trace(infile=infile)
    safe = batch_size#parser.safe * batch_size
    train = data[:safe].reshape(-1, batch_size, data.shape[1])[0]
    test = data[safe:-1, :].reshape(-1, batch_size, data.shape[1])
    all_residuals = np.empty((2, 27, len(test)))
    for i, sensor_type in enumerate(parser.type):
        indices_used = np.arange(parser.type_indices[i], parser.type_indices[i+1])
        num_sensors_used = len(indices_used)
        model = LinearRegression()
        train_used = train[:, indices_used]
        test_used = test[:, :, indices_used]
        for j in range(num_sensors_used):
            X_train, Y_train = X_Y_split(train_used, j, axis=1)
            X_test, Y_test = X_Y_split(test_used, j, axis=2)
            print(f"SENSOR {j}")
            for k in range(len(test_used)):
                X = X_test[k]
                y = Y_test[k]
                model.fit(X_train, Y_train)
                predictions = model.predict(X)
                residuals = np.abs(predictions - y)
                all_residuals[i, j, k] = residuals.mean()
                X_train = np.vstack((X_train, X))
                Y_train = np.hstack((Y_train, y))
    np.save("lr_residuals.npy", all_residuals)

anom_types = ["ramp", "gauss", "spike"]#, "ramp"]
NUM_ANOM_TYPES = len(anom_types)
# anomaly_sizes = (20, 2, 30, 40)
anomaly_sizes = (8, 12, 4)
np.random.seed(42)

def apply_anomaly(dataset, anom_type, uniform_size=1, gauss_size=1, spike_size=3, ramp_size=2):
    if anom_type == "gauss":
        anom = np.random.normal(0, gauss_size * dataset.std(), dataset.shape)
        return dataset + anom
    elif anom_type == "uniform":
        anom = dataset.std() * uniform_size
        return dataset + anom
    elif anom_type == "spike":
        spike_size = dataset.std() * spike_size
        for i in range(dataset.shape[0]):
            spike_index = np.random.choice(dataset.shape[1])
            dataset[i, spike_index] += spike_size
        return dataset
    elif anom_type == "ramp":
        N = dataset.shape[1]
        ramp = ramp_size * dataset.std() * ((np.arange(N) + 1) / N)
        # print(ramp)
        # print(dataset.std())
        return dataset + ramp
    else:
        raise ValueError(f"Invalid anomaly type: '{anom_type}'")
        

def testing_3(parser):
    infile = parser.infile
    batch_size = parser.batch
    with open(infile, "r") as file:
        raw_data = file.readlines()
    data = preprocess_trace(infile=infile)
    warmup_1_time = int(config["WARMUP_1_PROPORTION"] * parser.safe)
    warmup_2_time = parser.safe - warmup_1_time
    anomaly_sizes = np.zeros((11, 2))
    for spike_size in np.arange(1, 20, 1):
        for i, sensor_type in enumerate(parser.type):
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
            type_anomaly_times = []
            for sensor_index in range(num_sensors_used):
                anomaly_times = []
                X_train_lr, Y_train_lr = X_Y_split(train_lr, sensor_index)
                X_train_stl, Y_train_stl = X_Y_split(train_stl, sensor_index)
                # model.set_sensor_index(sensor_index)
                model = LargeWeightsRegressor()
                model.fit(X_train_lr, Y_train_lr)
                predictions = model.predict(X_train_stl)
                residuals = np.abs(predictions - Y_train_stl)
                residuals = cut(residuals, batch_size)
                residuals = residuals.reshape(-1, batch_size)
                formula = positive_synth(residuals, operators=parser.stl)
                # print(f"Sensor {sensor_index+1} formula: {formula}")
                X_train = np.vstack((X_train_lr, X_train_stl))
                Y_train = np.hstack((Y_train_lr, Y_train_stl))
                X_test, Y_test = X_Y_split(test, sensor_index)
                X_test = cut(X_test, batch_size)
                Y_test = cut(Y_test, batch_size)
                X_test = X_test.reshape(-1, batch_size, num_sensors_used - 1)
                Y_test = Y_test.reshape(-1, batch_size)
                # print("Anomaly size:", anomaly_size * anomaly_coef)
                Y_test = apply_anomaly(Y_test, "spike", spike_size=spike_size)
                num_anomalies = 0
                index = w_1 + w_2
                for X, Y in zip(X_test, Y_test):
                    new_batch = raw_data[index:index+batch_size]
                    start = max(0, len(Y_train) - 30 * batch_size)
                    X_train_window = X_train[start:]
                    Y_train_window = Y_train[start:]
                    model.fit(X_train_window, Y_train_window) # refit model
                    batch_residuals = np.abs(model.predict(X) - Y)
                    if not new_batch_ok(batch_residuals, formula, new_batch=new_batch, sensor_index=sensor_index, sensor_type=sensor_type, print_info=False):
                        num_anomalies += 1
                        # if not config["ADD_ANOMALIES_OFFLINE"]:
                        #     formula = positive_synth(residuals, operators=parser.stl)
                    else:
                        residuals = np.vstack((residuals, batch_residuals))
                        X_train = np.vstack((X_train, X))
                        Y_train = np.hstack((Y_train, Y))
                    index += batch_size
                print(f"{num_anomalies}/{len(Y_test)} anomalies detected")
                total_num_anomalies += num_anomalies
                num_total += len(Y_test)
                type_anomaly_times.append(anomaly_times)
            print("=" * 65)
            print(f"For type {sensor_type}, spike size {spike_size}")
            print(f"{total_num_anomalies}/{num_total} anomalies detected")
            print(f"Anomaly detection rate: {total_num_anomalies/num_total}")
            print("=" * 65)
    print(anomaly_sizes.tolist())


def testing_4(parser):
    from tree.tree import TreeNode
    infile = parser.infile
    batch_size = parser.batch
    with open(infile, "r") as file:
        raw_data = file.readlines()
    data = preprocess_trace(infile=infile)
    warmup_1_time = int(config["WARMUP_1_PROPORTION"] * parser.safe)
    warmup_2_time = parser.safe - warmup_1_time
    for train_tree_size in np.arange(72, 84, 12):
        confusion_matrix = np.zeros((NUM_ANOM_TYPES, NUM_ANOM_TYPES), dtype=int)
        for i, sensor_type in enumerate(parser.type):
            indices_used = np.arange(parser.type_indices[i], parser.type_indices[i+1])
            data_used = data[:, indices_used]
            w_1 = warmup_1_time * batch_size
            w_2 = warmup_2_time * batch_size
            train_lr = data_used[:w_1, :]
            train_stl = data_used[w_1:w_1+w_2, :]
            test = data_used[w_1 + w_2:, :]
            num_sensors_used = len(indices_used)
            for sensor_index in range(num_sensors_used):
                print(f"SENSOR {sensor_index}")
                X_train_lr, Y_train_lr = X_Y_split(train_lr, sensor_index)
                X_train_stl, Y_train_stl = X_Y_split(train_stl, sensor_index)
                model = LargeWeightsRegressor()
                X_train = np.vstack((X_train_lr, X_train_stl))
                Y_train = np.hstack((Y_train_lr, Y_train_stl))
                X_test, Y_test = X_Y_split(test, sensor_index)
                X_test = cut(X_test, batch_size)
                Y_test = cut(Y_test, batch_size)
                X_test = X_test.reshape(-1, batch_size, num_sensors_used - 1)
                Y_test = Y_test.reshape(-1, batch_size)
                model.fit(X_train, Y_train)
                set_size = train_tree_size // NUM_ANOM_TYPES
                X_train_tree = X_test[:train_tree_size]
                Y_train_tree = Y_test[:train_tree_size]
                X_test_tree = X_test[train_tree_size:]
                Y_test_tree = Y_test[train_tree_size:]
                indices = np.arange(train_tree_size)
                np.random.shuffle(indices)
                # for i in range(NUM_ANOM_TYPES):
                #     selection_indices = indices[i * set_size: (i + 1) * set_size]
                #     Y_train_tree[selection_indices] = apply_anomaly(Y_train_tree[selection_indices], anom_types[i], *anomaly_sizes)
                for i in range(NUM_ANOM_TYPES):
                    selection = slice(i * set_size, (i+1) * set_size)
                    Y_train_tree[selection] = apply_anomaly(Y_train_tree[selection], anom_types[i], *anomaly_sizes) 
                residuals = []
                for X, Y in zip(X_train_tree, Y_train_tree):
                    predictions = model.predict(X)
                    residuals.append(np.abs(predictions - Y))
                    X_train = np.vstack((X_train, X))
                    Y_train = np.hstack((Y_train, Y))
                    while X_train.shape[0] > 30:
                        X_train = X_train[1:]
                        Y_train = Y_train[1:]
                    model.fit(X_train, Y_train)
                residuals = np.array(residuals)
                labels = np.array([[anom_types[i]] * set_size for i in range(NUM_ANOM_TYPES)]).flatten()
                labelled_residuals = np.hstack((residuals, labels.reshape(train_tree_size, 1)))
                tree = TreeNode.build_tree(labelled_residuals, batch_size=batch_size, max_depth=5, binary=False, operators=parser.stl)
                old_cm = confusion_matrix.copy()
                # tree.print_tree()
                for i, anom_type in enumerate(anom_types):
                    for X, Y in zip(X_test_tree, Y_test_tree):
                        Y = apply_anomaly(Y.reshape(1, -1), anom_type, *anomaly_sizes).reshape(-1)
                        predictions = model.predict(X)
                        residuals = np.abs(predictions - Y)
                        predicted_label = tree.classify(residuals)
                        predicted_label_index = anom_types.index(predicted_label)
                        confusion_matrix[i, predicted_label_index] += 1
                        X_train = np.vstack((X_train, X))
                        Y_train = np.hstack((Y_train, Y))
                        while X_train.shape[0] > 30:
                            X_train = X_train[1:]
                            Y_train = Y_train[1:]
                        model.fit(X_train, Y_train)
                cm = confusion_matrix - old_cm
                print(cm)
                print(f"Anomaly types: {anom_types}")
                accuracy = cm.trace() / cm.sum()
                print("Accuracy:", accuracy)
                if accuracy > 0.98:
                    tree.print_tree()
                # if accuracy < 0.7:
                #     tree.print_tree()
                #     exit()
                # exit()
        print("Training size was:", train_tree_size)
        print(confusion_matrix.tolist())
        print("Accuracy:", confusion_matrix.trace() / confusion_matrix.sum())

def testing_5(parser):
    from tree.bin_class import build, update
    infile = parser.infile
    batch_size = parser.batch
    with open(infile, "r") as file:
        raw_data = file.readlines()
    data = preprocess_trace(infile=infile)
    # warmup_1_time = int(config["WARMUP_1_PROPORTION"] * parser.safe)
    # warmup_2_time = parser.safe - warmup_1_time
    # sizes = np.arange(6, 54, 12)
    sizes = [90]
    for train_tree_size in sizes:
        confusion_matrix = np.zeros((2,2))
        for i, sensor_type in enumerate(parser.type):
            indices_used = np.arange(parser.type_indices[i], parser.type_indices[i+1])
            data_used = data[:, indices_used]
            w_1 = 2 * batch_size#warmup_1_time * batch_size
            w_2 = 18 * batch_size #warmup_2_time * batch_size
            train_lr = data_used[:w_1, :]
            train_stl = data_used[w_1:w_1+w_2, :]
            test = data_used[w_1 + w_2:, :]
            num_sensors_used = len(indices_used)
            for sensor_index in range(num_sensors_used):
                print(f"SENSOR {sensor_index}")
                X_train_lr, Y_train_lr = X_Y_split(train_lr, sensor_index)
                X_train_stl, Y_train_stl = X_Y_split(train_stl, sensor_index)
                model = LargeWeightsRegressor()
                model.fit(X_train_lr, Y_train_lr)
                initial_residuals = np.abs(model.predict(X_train_stl) - Y_train_stl).reshape(-1, 96)
                X_train = np.vstack((X_train_lr, X_train_stl))
                Y_train = np.hstack((Y_train_lr, Y_train_stl))
                model.fit(X_train, Y_train)
                X_test, Y_test = X_Y_split(test, sensor_index)
                X_test = cut(X_test, batch_size)
                Y_test = cut(Y_test, batch_size)
                X_test = X_test.reshape(-1, batch_size, num_sensors_used - 1)
                Y_test = Y_test.reshape(-1, batch_size)
                # train_tree_size = DEFINED ABOVE
                set_size = train_tree_size // NUM_ANOM_TYPES
                X_train_tree = X_test[:train_tree_size]
                Y_train_tree = Y_test[:train_tree_size]
                X_test_tree = X_test[train_tree_size:]
                Y_test_tree = Y_test[train_tree_size:]
                for i in range(NUM_ANOM_TYPES):
                    selection = slice(i * set_size, (i+1) * set_size)
                    Y_train_tree[selection] = apply_anomaly(Y_train_tree[selection], anom_types[i], *anomaly_sizes) 
                residuals = []
                for X, Y in zip(X_train_tree, Y_train_tree):
                    predictions = model.predict(X)
                    residuals.append(np.abs(predictions - Y))
                    X_train = np.vstack((X_train, X))
                    Y_train = np.hstack((Y_train, Y))
                    while X_train.shape[0] > 30:
                        X_train = X_train[1:]
                        Y_train = Y_train[1:]
                    model.fit(X_train, Y_train)
                pos_residuals = np.array(residuals)
                neg_residuals = initial_residuals
                tree = build(neg_train=neg_residuals, pos_train=pos_residuals, operators=parser.stl)
                # tree.print_tree()
                classes = ["Safe", "Anomaly"]
                # anom_types = ["uniform"]
                index = 0
                for i, c in enumerate(classes):
                    for X, Y in zip(X_test_tree, Y_test_tree):
                        if c == "Anomaly":
                            anom_type = anom_types[index % NUM_ANOM_TYPES]
                            Y = apply_anomaly(Y.reshape(1, -1), anom_type, *anomaly_sizes).reshape(-1)
                        index += 1
                        predictions = model.predict(X)
                        residuals = np.abs(predictions - Y)
                        predicted_label = tree.classify(residuals)
                        predicted_label_index = classes.index(predicted_label)
                        confusion_matrix[i, predicted_label_index] += 1
                        X_train = np.vstack((X_train, X))
                        Y_train = np.hstack((Y_train, Y))
                        while X_train.shape[0] > 30:
                            X_train = X_train[1:]
                            Y_train = Y_train[1:]
                        model.fit(X_train, Y_train)
        print("Train size:", train_tree_size)
        print(confusion_matrix.tolist())
        print("Accuracy:", confusion_matrix.trace() / confusion_matrix.sum())

def testing_6(parser):
    infile = parser.infile
    batch_size = parser.batch
    with open(infile, "r") as file:
        raw_data = file.readlines()
    data = preprocess_trace(infile=infile)
    # warmup_1_time = int(config["WARMUP_1_PROPORTION"] * parser.safe)
    # warmup_2_time = parser.safe - warmup_1_time
    confusion_matrix = np.zeros((2,2))
    for i, sensor_type in enumerate(parser.type):
        indices_used = np.arange(parser.type_indices[i], parser.type_indices[i+1])
        data_used = data[:, indices_used]
        w_1 = 10 * batch_size#warmup_1_time * batch_size
        w_2 = 30 * batch_size #warmup_2_time * batch_size
        train_lr = data_used[:w_1, :]
        train_stl = data_used[w_1:w_1+w_2, :]
        test = data_used[w_1 + w_2:, :]
        num_sensors_used = len(indices_used)
        for sensor_index in range(num_sensors_used):
            print(f"SENSOR {sensor_index}")
            X_train_lr, Y_train_lr = X_Y_split(train_lr, sensor_index)
            X_train_stl, Y_train_stl = X_Y_split(train_stl, sensor_index)
            model = LargeWeightsRegressor()
            model.fit(X_train_lr, Y_train_lr)
            initial_residuals = np.abs(model.predict(X_train_stl) - Y_train_stl).reshape(-1, 96)
            X_train = np.vstack((X_train_lr, X_train_stl))
            Y_train = np.hstack((Y_train_lr, Y_train_stl))
            model.fit(X_train, Y_train)
            X_test, Y_test = X_Y_split(test, sensor_index)
            X_test = cut(X_test, batch_size)
            Y_test = cut(Y_test, batch_size)
            X_test = X_test.reshape(-1, batch_size, num_sensors_used - 1)
            Y_test = Y_test.reshape(-1, batch_size)
            # train_tree_size = DEFINED ABOVE
            train_tree_size = 60
            X_train_tree = X_test[:train_tree_size]
            Y_train_tree = Y_test[:train_tree_size]
            X_test_tree = X_test[train_tree_size:]
            Y_test_tree = Y_test[train_tree_size:]
            set_size = train_tree_size // NUM_ANOM_TYPES
            # Y_train_tree = apply_anomaly(Y_train_tree, "uniform", uniform_size=100)
            residuals = []
            for X, Y in zip(X_train_tree, Y_train_tree):
                predictions = model.predict(X)
                residuals.append(np.abs(predictions - Y))
                X_train = np.vstack((X_train, X))
                Y_train = np.hstack((Y_train, Y))
                while X_train.shape[0] > 30:
                    X_train = X_train[1:]
                    Y_train = Y_train[1:]
                model.fit(X_train, Y_train)
            neg_residuals = initial_residuals
            formula = positive_synth(neg_residuals, operators=parser.stl)
            classes = ["Safe", "Anomaly"]
            # anom_types = ["uniform"]
            index = 0
            for i, c in enumerate(classes):
                for X, Y in zip(X_test_tree, Y_test_tree):
                    if c == "Anomaly":
                        anom_type = anom_types[index % NUM_ANOM_TYPES]
                        Y = apply_anomaly(Y.reshape(1, -1), anom_type, *anomaly_sizes).reshape(-1)
                    index += 1
                    predictions = model.predict(X)
                    residuals = np.abs(predictions - Y)
                    rob = formula.evaluate(residuals.reshape(1, -1), return_arr=True)
                    predicted_label = "Anomaly" if rob.min() < 0 else "Safe"
                    predicted_label_index = classes.index(predicted_label)
                    confusion_matrix[i, predicted_label_index] += 1
                    X_train = np.vstack((X_train, X))
                    Y_train = np.hstack((Y_train, Y))
                    if True:
                        while X_train.shape[0] > 30:
                            X_train = X_train[1:]
                            Y_train = Y_train[1:]
                        model.fit(X_train, Y_train)
    print(confusion_matrix.tolist())
    print("Accuracy:", confusion_matrix.trace() / confusion_matrix.sum())

if __name__ == "__main__":
    from parser import Parser
    parser = Parser()
    parser.parse("spec.file")
    offline_ad(parser)