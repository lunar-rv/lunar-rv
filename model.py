"""
REQUIREMENTS:
    1. Linear regression weights and biases are updated with every new reading
    2. Generate STL spec based on typical behaviour of the system
    3. Flag anomalous behaviour
    4. Human engineer classifies anomalous behaviour
"""

""" 

    BY THURSDAY:
        - Prepare a presentation
        - Machine learning model for comparison
        - Integrate the semi-supervised anomaly detection
        - Option to only allow invariant operators

"""

##### IMPORTS #####
from preproc import preprocess
from sklearn.linear_model import LinearRegression
import numpy as np
from synth import positive_synth
from graphs import plot_array
from tree.tree import TreeNode
from tree.bin_class import build, update
import warnings
from file_io import write_new_batch
import json

with open('config.json', 'r') as file:
    config = json.load(file)

warnings.filterwarnings(action="ignore", message="genfromtxt: Empty input file")

def apply_anomaly(
    dataset: np.ndarray, anomaly_type: str, sensor_index: int
) -> np.ndarray:
    if anomaly_type == "1":  # single sensor
        dataset[:, sensor_index] += config["ANOMALY_SIZE"]
    elif anomaly_type == "2":
        dataset[:, sensor_index] += 5 * config["ANOMALY_SIZE"]
    elif anomaly_type == "3":
        dataset += config["ANOMALY_SIZE"]
    else:
        raise ValueError(f"Unrecognised anomaly type: '{anomaly_type}'")
    return dataset


def get_residuals(
    new_batch, safe_trace_file=config["SAFE_TRACE_FILE"], sensor_index=0, anomaly_type=None
) -> None:
    def X_Y_split(data: np.ndarray, i: int):
        X = np.delete(data, i, axis=1)
        Y = data[:, i].astype(float)
        return X, Y
    train = preprocess(safe_trace_file)
    test = preprocess("".join(new_batch), csv=False)
    if anomaly_type is not None:
        test = apply_anomaly(test, anomaly_type, sensor_index)
    X_train, Y_train = X_Y_split(train, sensor_index)
    X_test, Y_test = X_Y_split(test, sensor_index)
    MODEL = eval(config["MODEL"])
    MODEL.fit(X_train, Y_train)
    predictions = MODEL.predict(X_test)
    abs_residuals = np.abs(predictions - Y_test) * 1000
    return abs_residuals

def new_batch_ok(residuals, formula=None) -> bool:
    if formula:
        classification = formula.evaluate(residuals.reshape(1, -1), labels=False)
        if classification[0] < 0:
            return False
    return True

def update_spec(
    spec_file=config["SPEC_FILE"],
    residuals_file=config["RESIDUALS_FILE"],
    anomalies_file=config["ANOMALIES_FILE"],
    operators=config["OPERATORS"],
    invariance=config["INVARIANCE"],
    bin_classifier=None,
    new_trace=None,
    new_label=None,
) -> tuple:
    negative_traces = np.genfromtxt(residuals_file, delimiter=",", dtype=float)
    positive_traces = np.genfromtxt(anomalies_file, delimiter=",")
    if len(positive_traces) < config["WARMUP_ANOMALIES"] or positive_traces.ndim == 1:
        spec = positive_synth(operators=operators, traces=negative_traces[:, np.newaxis, :], invariance=invariance)
    elif len(positive_traces) == config["WARMUP_ANOMALIES"]:
        positive_values = positive_traces[:, :-1].astype(float)
        bin_classifier = build(negative_traces, positive_values, invariance=invariance)
        spec = bin_classifier.formula
    else:
        bin_classifier = update(bin_classifier, new_trace, new_label, invariance=invariance)
        spec = bin_classifier.formula
    with open(spec_file, "r+") as s:
        old_spec = s.read()
        if old_spec == repr(spec):
            return spec, bin_classifier
        s.seek(0)
        s.write(repr(spec))
        s.truncate()
    print("=" * 50)
    print(f"Formula is now: {spec}")
    print("=" * 50)
    return spec, bin_classifier


def log_anomaly(
    batch, trace, sensor_index, tree=None, grow_tree=False, plot=True
) -> TreeNode:
    raw_data = preprocess(
        "".join(batch), csv=False, time_features=False, season_features=False
    )
    sensor_values = raw_data[:, sensor_index]
    trace_np = np.array(trace.split(",")).astype(float)
    if tree:
        prediction = tree.classify(trace_np)
        print("Predicted anomaly type:", prediction)
    if config["PLOT_ANOMALY_GRAPHS"]:
        plot_array(
            trace=sensor_values, sensor_index=sensor_index, keyword="Real Sensor Values"
        )
        plot_array(
            trace=trace_np, sensor_index=sensor_index, keyword="Magnitude of Residuals"
        )
    prompt = "Enter anomaly type:\n - Press Enter if unknown\n - Type 'safe' if this is a false alarm\n>"
    response = input(prompt)
    anomaly_type = response if response else "unknown"
    if anomaly_type.lower() == "safe":
        with open(config["RESIDUALS_FILE"], "a") as r:
            r.write("\n" + trace)
        write_new_batch(batch, config["SAFE_TRACE_FILE"])
        return tree
    with open(config["ANOMALIES_FILE"], "a") as a:
        a.write("\n" + trace + "," + anomaly_type)
    if anomaly_type == "unknown":
        return tree
    if grow_tree:
        print("Building tree...", flush=True)
        tree = TreeNode.build_tree(
            np.append(trace_np, anomaly_type).reshape(1, -1), binary=False, max_depth=5
        )
        print("Tree built!")
    else:
        print("Updating tree...", flush=True)
        tree.update_tree(np.append(trace_np, anomaly_type), binary=False)
        tree.print_tree()
        print("\nPress Enter to continue (or q to quit) ")
    return tree


