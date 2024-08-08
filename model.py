from preproc import preprocess
import numpy as np
from synth import positive_synth
from graphs import plot_array
from tree.tree import TreeNode
from tree.bin_class import build, update
import warnings
from file_io import write_new_batch, write_weights, get_filename
import json
from scipy import stats
from regressor import LargeWeightsRegressor
from ui import print_anomaly_info, get_and_display_anomaly_times
import os

model = LargeWeightsRegressor()
with open('config.json', 'r') as file:
    config = json.load(file)

warnings.filterwarnings(action="ignore", message="genfromtxt: Empty input file")

def get_residuals(
    train, test, sensor_index, sensor_type
) -> None:
    def X_Y_split(data: np.ndarray, i: int):
        X = np.delete(data, i, axis=1)
        Y = data[:, i].astype(float)
        return X, Y
    X_train, Y_train = X_Y_split(train, sensor_index)
    X_test, Y_test = X_Y_split(test, sensor_index)
    model.set_sensor_index(sensor_index)
    model.fit(X_train, Y_train)
    write_weights(model, sensor_type=sensor_type)
    predictions = model.predict(X_test)
    abs_residuals = np.abs(predictions - Y_test)
    if sensor_type == "PRESSURE":
        abs_residuals *= 1000 # mBar
    return abs_residuals

def get_safety_prob(sensor_index, mean_rob: float, sensor_type=None) -> float:
    safe_residuals_file = get_filename("residuals", sensor_index, sensor_type=sensor_type)
    safe_traces = np.genfromtxt(safe_residuals_file, delimiter=",", dtype=float)
    sigma = np.std(safe_traces)
    mu = np.mean(safe_traces)
    return stats.norm.cdf(mean_rob, mu, sigma)

def new_batch_ok(residuals, formula=None, new_batch: list = None, sensor_index: int = None, sensor_type: str = None) -> bool:
    if formula:
        evaluation = formula.evaluate_single(residuals, labels=False)
        ### NEED TO DEAL WITH PRINTING ANOMALY INFO, BUT DO THIS LATER
        # safety_prob = get_safety_prob(sensor_index=sensor_index, mean_rob=classification, sensor_type=sensor_type)
        # rounded_rob = np.round(classification, 4)
        # print("Robustness: ", rounded_rob)
        # print(f"Likelihood of robustness {rounded_rob} or lower: {np.round(safety_prob, 4)}")
        # print(f"Minimum threshold: {np.round(get_safety_prob(sensor_index=sensor_index, mean_rob=0, sensor_type=sensor_type), 4)}")
        if evaluation.min() < 0:
            anomaly_start_indices = np.where(evaluation < 0)[0].tolist()
            bounds = get_and_display_anomaly_times(anomaly_start_indices, formula, new_batch)
            if formula.last is not None:
                bounds = np.array(bounds) + 2 * formula.last.size
                residuals = np.hstack((formula.last.flatten(), residuals))
            #### ADD formula.last TO START OF RESIDUALS ####
            ### Problem is that I haven't stored the raw sensor values in 'last' ###
            ### Could extract these values from a file somewhere ###
            if config["PLOT_ANOMALY_GRAPHS"]:
                # print(anomaly_start_indices)
                # print(residuals.tolist())
                # exit()
                raw_data = preprocess("".join(new_batch), csv=False)
                sensor_values = raw_data[:, sensor_index]
                plot_array(
                    trace=sensor_values, sensor_index=sensor_index, keyword="Real Sensor Values"
                )
                plot_array(
                    trace=residuals, 
                    sensor_index=sensor_index, 
                    keyword="Magnitude of Residuals", 
                    boundary=formula.boundary, 
                    bounds=bounds
                )

            # print_anomaly_info(model, new_batch, formula, sensor_type)
            return False
    return True

def update_spec(
    sensor_index,
    operators=config["OPERATORS"],
    invariance=config["INVARIANCE"],
    bin_classifier=None,
    new_trace=None,
    new_label=None,
    formulae=[],
    sensor_type=None,
) -> tuple:
    spec_file = get_filename("specs", sensor_index, suffix=".stl", remove_plural=True, sensor_type=sensor_type)
    residuals_file = get_filename("residuals", sensor_index, sensor_type=sensor_type)
    anomalies_file = get_filename("anomalies", sensor_index, sensor_type=sensor_type)
    negative_traces = np.genfromtxt(residuals_file, delimiter=",", dtype=float)
    positive_traces = np.genfromtxt(anomalies_file, delimiter=",")
    use_mean = config["USE_MEAN"]
    if len(positive_traces) < config["WARMUP_ANOMALIES"] or positive_traces.ndim == 1:
        spec = positive_synth(operator="F", traces=negative_traces)
    elif len(positive_traces) == config["WARMUP_ANOMALIES"]:
        positive_values = positive_traces[:, :-1].astype(float)
        bin_classifier = build(negative_traces, positive_values, invariance=invariance, use_mean=use_mean)
        spec = bin_classifier.formula
    else:
        bin_classifier = update(bin_classifier, new_trace, new_label, invariance=invariance, use_mean=use_mean)
        spec = bin_classifier.formula
    formulae[sensor_index] = spec
    if not os.path.exists(spec_file):
        with open(spec_file, "w"):
            pass
    with open(spec_file, "r+") as s:
        old_spec = s.read()
        if old_spec == repr(spec):
            return formulae, bin_classifier
        s.seek(0)
        s.write(repr(spec))
        s.truncate()
    print("=" * 50)
    print(f"Formula is now: {spec}")
    print("=" * 50)
    return formulae, bin_classifier


def log_anomaly(
    batch, trace, sensor_index, tree=None, warmup2=False, sensor_type=None
) -> TreeNode:
    raw_data = preprocess(
        "".join(batch), csv=False, time_features=False, season_features=False
    )
    trace_np = np.array(trace.split(",")).astype(float)
    if tree:
        prediction = tree.classify(trace_np)
        print("Predicted anomaly type:", prediction)
    prompt = "Enter anomaly type:\n - Press Enter if unknown\n - Type 'safe' if this is a false alarm\n>"
    response = input(prompt)
    anomaly_type = response if response else "unknown"
    anomalies_file = get_filename("anomalies", sensor_index, sensor_type=sensor_type)
    residuals_file = get_filename("residuals", sensor_index, sensor_type=sensor_type)
    if anomaly_type.lower() == "safe":
        with open(residuals_file, "a") as r:
            r.write("\n" + trace)
        write_new_batch(batch, config["SAFE_TRACE_FILE"])
        return False, tree
    with open(anomalies_file, "a") as a:
        a.write("\n" + trace + "," + anomaly_type)
    if anomaly_type == "unknown":
        return True, tree
    if not warmup2 and not tree:
        print("Building tree...", flush=True)
        tree = TreeNode.build_tree(
            np.append(trace_np, anomaly_type).reshape(1, -1), binary=False, max_depth=5
        )
        print("Tree built!")
    else:
        tree.update_tree(np.append(trace_np, anomaly_type), binary=False)
        tree.print_tree()
    return True, tree

def apply_anomaly(data: np.ndarray, anomaly_indices: np.ndarray, anom_type: str) -> np.ndarray:
    if anom_type != "normal":
        pressure_indices = anomaly_indices[anomaly_indices < 27]
        temp_indices = anomaly_indices[anomaly_indices >= 27]
        data[:, pressure_indices] += config[f"{anom_type.upper()}_PRESSURE_ANOMALY_SIZE"]
        data[:, temp_indices] += config[f"{anom_type.upper()}_TEMPERATURE_ANOMALY_SIZE"]
    return data

def main():
    residuals_1 = np.genfromtxt("outputs/residuals/sensor_1_residuals.csv", delimiter=",", dtype=float)
    residuals_2 = np.genfromtxt("outputs/residuals/sensor_2_residuals.csv", delimiter=",", dtype=float)
    print(np.mean(residuals_1, axis=1).max())
    print(np.mean(residuals_2, axis=1).max())



if __name__ == "__main__":
    main()