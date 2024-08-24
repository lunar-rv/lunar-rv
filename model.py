from preproc import preprocess_trace
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
from ui import print_anomaly_info, get_and_display_anomaly_times, get_time_period
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
    return abs_residuals

def get_safety_dist(sensor_index, sensor_type=None) -> float:
    safe_residuals_file = get_filename("residuals", sensor_index, sensor_type=sensor_type)
    safe_traces = np.genfromtxt(safe_residuals_file, delimiter=",", dtype=float)
    sigma = np.std(safe_traces)
    mu = np.mean(safe_traces)
    return mu, sigma

def new_batch_ok(residuals, formula=None, new_batch: list = None, sensor_index: int = None, sensor_type: str = None) -> bool:
    time_period = get_time_period(new_batch)
    if formula is None:
        return True
    raw_data = preprocess_trace(new_batch=new_batch)
    sensor_values = raw_data[:, sensor_index]
    backlog_size: int = formula.last_residuals.size if formula.last_residuals is not None else 0
    if backlog_size != 0:
        old_residuals = np.hstack((formula.last_residuals.flatten(), residuals))
        old_sensor_values = np.hstack((formula.last_raw_values.flatten(), sensor_values))
    else:
        old_residuals = residuals
        old_sensor_values = sensor_values
    evaluations = formula.evaluate_single(residuals, labels=False, raw_values=sensor_values)
    rob = np.hstack([rho.flatten() for rho in evaluations.values()])
    if rob.min() >= 0:
        return True
        # print("Evaluation", evaluation)
    print("Failed to satisfy formula: ", formula)
    mean_res = np.round(residuals.mean(), 4)
    # # min_rob = np.round(rob.min(), 4)
    mu, sigma = get_safety_dist(sensor_index, sensor_type)
    mean_safety_prob = np.round(stats.norm.cdf(mean_res, mu, sigma), 4)
    # min_safety_prob = np.round(stats.norm.cdf(residuals.max(), mu, sigma), 4)
    # print("Average residuals:", residuals.mean())
    # print("Minimum residual:", residuals.max())
    print(f"Likelihood of average residuals of {mean_res} or higher: {1-mean_safety_prob}")
    # print(f"Likelihood of robustness {min_rob} or lower: {min_safety_prob}")
    if config["PLOT_ANOMALY_GRAPHS"]:
        for i in range(3):
            phi = formula[i]
            this_evaluation = evaluations[phi][0]#[:shortest_length]
            if this_evaluation.min() >= 0:
                continue
            anomaly_start_indices = np.where(this_evaluation < 0)[0].tolist()
            end = phi.end if phi.end is not None else formula.max_length
            bounds, batch_start_time = get_and_display_anomaly_times(anomaly_start_indices, formula, new_batch, prev_backlog_size=backlog_size, end=end)
            bounds = np.array(bounds) + backlog_size
            plot_array(
                trace=old_sensor_values,
                sensor_index=sensor_index,
                keyword="Actual Sensor Values",
                bounds=bounds,
                time_period=time_period,
                sensor_type=sensor_type,
                batch_start_time=batch_start_time,
                backlog_size=backlog_size,
            )
            plot_array(
                trace=old_residuals, 
                sensor_index=sensor_index, 
                keyword="Magnitude of Residuals", 
                formula=phi,
                bounds=bounds,
                time_period=time_period,
                batch_start_time=batch_start_time,
                backlog_size=backlog_size,
                sensor_type=sensor_type,
            )
    print_anomaly_info(model, new_batch, formula)
    return False

def update_spec(
    sensor_index,
    operators,
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

    if len(positive_traces) < config["WARMUP_ANOMALIES"] or positive_traces.ndim == 1:
        prev_formula = formulae[sensor_index] if len(formulae) > sensor_index else None
        spec = positive_synth(traces=negative_traces, prev_formula=prev_formula, operators=operators)
    elif len(positive_traces) == config["WARMUP_ANOMALIES"]:
        positive_values = positive_traces[:, :-1].astype(float)
        bin_classifier = build(negative_traces, positive_values)
        spec = bin_classifier.formula
    else:
        bin_classifier = update(bin_classifier, new_trace, new_label)
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
    trace, sensor_index, tree=None, warmup2=False, sensor_type=None, 
) -> TreeNode:
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
        return False, tree
    with open(anomalies_file, "a") as a:
        a.write("\n" + trace + "," + anomaly_type)
    if anomaly_type == "unknown":
        return True, tree
    if not warmup2 and not tree:
        print("Building tree...", flush=True)
        tree = TreeNode.build_tree(
            np.append(trace_np, anomaly_type).reshape(1, -1), batch_size=trace_np.size, binary=False, max_depth=5
        )
        print("Tree built!")
    else:
        tree.update_tree(trace=np.append(trace_np, anomaly_type), batch_size=trace_np.size, binary=False)
        tree.print_tree()
    return True, tree

    # "SMALL_PRESSURE_ANOMALY_SIZE": 0.001,
    # "LARGE_PRESSURE_ANOMALY_SIZE": 0.005,
    # "SMALL_TEMPERATURE_ANOMALY_SIZE": 0.5,
    # "LARGE_TEMPERATURE_ANOMALY_SIZE": 2.5,

def apply_anomaly(data: np.ndarray, anomaly_indices: np.ndarray, anom_type: str) -> np.ndarray:
    if anom_type != "normal":
        # pressure_indices = anomaly_indices[anomaly_indices < 27]
        # temp_indices = anomaly_indices[anomaly_indices >= 27]
        # p_increase = 0.001 if anom_type == "small" else 0.005
        # t_increase = 0.5 if anom_type == "small" else 2.5
        # data[:, pressure_indices] += p_increase
        # data[:, temp_indices] += t_increase
        anomaly_size = data[anomaly_indices].std(axis=0)
        if anom_type == "large":
            anomaly_size *= 5
        data[:, anomaly_indices] += anomaly_size
    return data

def main():
    residuals_1 = np.genfromtxt("outputs/residuals/sensor_1_residuals.csv", delimiter=",", dtype=float)
    residuals_2 = np.genfromtxt("outputs/residuals/sensor_2_residuals.csv", delimiter=",", dtype=float)
    print(np.mean(residuals_1, axis=1).max())
    print(np.mean(residuals_2, axis=1).max())



if __name__ == "__main__":
    main()