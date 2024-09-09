
import json
from synth import positive_synth
from evaluation import cut, X_Y_split
from regressor import LargeWeightsRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
from preproc import preprocess_trace
from model import new_batch_ok#, apply_anomaly

with open("config.json") as config_file:
    config = json.load(config_file)

anom_types = ["ramp", "gauss", "spike"]#, "ramp"]
NUM_ANOM_TYPES = len(anom_types)
# anomaly_sizes = (20, 2, 30, 40)
anomaly_sizes = (1, 3, 2)
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
        
# Function to test the decision tree performance
def testing_1(parser):
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

# Function for testing the AD performance of binary classifier
def testing_2(parser):
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

# Function to test the AD performance
def testing_3(parser):
    infile = parser.infile
    batch_size = parser.batch
    with open(infile, "r") as file:
        raw_data = file.readlines()
    data = preprocess_trace(infile=infile)
    # warmup_1_time = int(config["WARMUP_1_PROPORTION"] * parser.safe)
    # warmup_2_time = parser.safe - warmup_1_time
    confusion_matrix = np.zeros((2,2))
    for i, sensor_type in enumerate(parser.type):
        if sensor_type != "temperature":
            continue
        indices_used = np.arange(parser.type_indices[i], parser.type_indices[i+1])
        data_used = data[:, indices_used]
        w_1 = 2 * batch_size#warmup_1_time * batch_size
        w_2 = 18 * batch_size #warmup_2_time * batch_size
        train_lr = data_used[:w_1, :]
        train_stl = data_used[w_1:w_1+w_2, :]
        test = data_used[w_1 + w_2:, :]
        num_sensors_used = len(indices_used)
        # anom_results = np.zeros((10))
        # for sensor_index in range(16,27):
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
            X_test_tree = X_test[:1]
            Y_test_tree = Y_test[:1]
            set_size = train_tree_size // NUM_ANOM_TYPES
            # Y_train_tree = apply_anomaly(Y_train_tree, "uniform", uniform_size=100)
            residuals = []
            # for X, Y in zip(X_train_tree, Y_train_tree):
            #     predictions = model.predict(X)
            #     residuals.append(np.abs(predictions - Y))
            #     X_train = np.vstack((X_train, X))
            #     Y_train = np.hstack((Y_train, Y))
            #     while X_train.shape[0] > 30:
            #         X_train = X_train[1:]
            #         Y_train = Y_train[1:]
            #     model.fit(X_train, Y_train)
            neg_residuals = initial_residuals
            formula = positive_synth(neg_residuals, operators=parser.stl)
            # print(formula)
            classes = ["Anomaly", "Safe"]
            # anom_types = ["uniform"]
            day_index = 0
            for i, c in enumerate(classes):
                for X, Y in zip(X_test, Y_test):
                    if c == "Anomaly":
                        anom_type = anom_types[day_index % NUM_ANOM_TYPES]
                        Y = apply_anomaly(Y.reshape(1, -1), anom_type, *anomaly_sizes).reshape(-1)
                    day_index += 1
                    predictions = model.predict(X)
                    residuals = np.abs(predictions - Y)
                    rob = formula.evaluate(residuals.reshape(1, -1), return_arr=True)
                    predicted_label = "Anomaly" if rob.min() < 0 else "Safe"
                    # anoms = np.array([9, 33, 82, 83, 84, 85, 86, 87, 88, 129]) - (w_1 + w_2) / batch_size
                    # # print(anoms)
                    # if day_index in anoms:
                    #     if predicted_label == "Anomaly":
                    #         anom_results[anoms.tolist().index(day_index)] += 1
                    #     print("index:", day_index + (w_1 + w_2) / batch_size)
                    #     print(predicted_label)
                    # if predicted_label == "Safe":
                    #     neg_residuals = np.vstack((neg_residuals, residuals))
                        # formula = positive_synth(neg_residuals, operators=parser.stl)
                    if predicted_label == "Anomaly" and c == "Safe":
                        neg_residuals = np.vstack((neg_residuals, residuals))
                        formula = positive_synth(neg_residuals, operators=parser.stl)
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
    # print(anom_results)

if __name__ == "__main__":
    from parser import Parser
    parser = Parser()
    parser.parse("spec.file")
    offline_ad(parser)