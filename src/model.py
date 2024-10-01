from preproc import preprocess_trace
import numpy as np
from synth import positive_synth
from graphs import plot_array
from tree.tree import TreeNode
import warnings
from file_io import write_weights, get_filename
import json
from scipy import stats
from regressor import LargeWeightsRegressor
from ui import get_and_display_anomaly_times, get_time_period
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
    residuals = predictions - Y_test
    return residuals

def get_safety_dist(sensor_index, sensor_type=None) -> float:
    safe_residuals_file = get_filename("residuals", sensor_index, sensor_type=sensor_type)
    safe_traces = np.abs(np.genfromtxt(safe_residuals_file, delimiter=",", dtype=float))
    sigma = np.std(safe_traces)
    mu = np.mean(safe_traces)
    return mu, sigma

def qualitative_rob(evaluations: dict, residuals: np.ndarray, backlog_size: int) -> bool:
    epsilon = 1e-6
    for key, value in evaluations.items():
        if value.min() < -epsilon:
            if key.name == "G":
                day_data = residuals[backlog_size:]
                if np.abs(day_data).max() <= key.boundary + epsilon:
                    continue
            return False
    return True

def new_batch_ok(residuals, start_index: int, formula=None, new_batch: list = None, sensor_index: int = None, sensor_type: str = None, print_info=True) -> bool:
    time_period = get_time_period(new_batch)
    if formula is None:
        return True
    raw_data = preprocess_trace(new_batch=new_batch)
    sensor_values = raw_data[:, start_index + sensor_index]
    backlog_size: int = formula.last_residuals.size if formula.last_residuals is not None else 0
    # print("R", residuals.shape, "LR", formula.last_residuals.shape if formula.last_residuals is not None else None)
    old_sensor_values = sensor_values.copy()
    old_residuals = residuals.copy()
    if backlog_size != 0:
        residuals = np.hstack((formula.last_residuals.flatten(), residuals))
        sensor_values = np.hstack((formula.last_raw_values.flatten(), sensor_values))
    evaluations = formula.evaluate_single(old_residuals, labels=False, raw_values=old_sensor_values)
    # for key, value in evaluations.items():
    #     print(key, value.shape)
    if qualitative_rob(evaluations, residuals=residuals, backlog_size=backlog_size):
        return True
    if print_info:
        print("Failed to satisfy formula: ", formula)
        mean_res = np.abs(np.round(residuals.mean(), 4))
        mu, sigma = get_safety_dist(sensor_index, sensor_type)
        mean_safety_prob = np.round(stats.norm.cdf(mean_res, mu, sigma), 4)
        print(f"Likelihood of average residual size of {mean_res} or higher: {1-mean_safety_prob}")
    if config["PLOT_ANOMALY_GRAPHS"]:
        for i in range(len(list(formula))):
            phi = formula[i]
            this_evaluation = evaluations[phi][0]
            if this_evaluation.min() >= 0:
                continue
            if phi.name == "G":
                if np.abs(old_residuals).max() <= phi.boundary:
                    continue
            # print("THIS_EVAL_SHAPE:", this_evaluation.shape)
            # print("THIS_EVAL:", this_evaluation)
            anomaly_start_indices = np.where(this_evaluation < 0)[0].tolist()
            # print("ASI:", anomaly_start_indices)
            end = phi.end if phi.end is not None else formula.max_length
            bounds, batch_start_time = get_and_display_anomaly_times(anomaly_start_indices, phi, new_batch, prev_backlog_size=backlog_size, end=end)
            if phi.name == "G":
                graph_size = len(residuals)
                bounds = [backlog_size, graph_size],
            preds = sensor_values + residuals
            plot_array(
                trace=sensor_values,
                sensor_index=sensor_index,
                keyword="Actual Sensor Values",
                bounds=bounds,
                time_period=time_period,
                sensor_type=sensor_type,
                batch_start_time=batch_start_time,
                backlog_size=backlog_size,
                preds=preds
            )
            plot_array(
                trace=np.abs(residuals), 
                sensor_index=sensor_index, 
                keyword="Magnitude of Residuals", 
                formula=phi,
                bounds=bounds,
                time_period=time_period,
                batch_start_time=batch_start_time,
                backlog_size=backlog_size,
                sensor_type=sensor_type,
                preds=None
            )
    return False

def update_spec(
    sensor_index,
    operators,
    formulae=[],
    sensor_type=None,
) -> tuple:
    spec_file = get_filename("specs", sensor_index, suffix=".stl", remove_plural=True, sensor_type=sensor_type)
    residuals_file = get_filename("residuals", sensor_index, sensor_type=sensor_type)
    positive_traces = np.genfromtxt(residuals_file, delimiter=",", dtype=float)
    prev_formula = formulae[sensor_index] if len(formulae) > sensor_index else None
    spec = positive_synth(traces=positive_traces, prev_formula=prev_formula, operators=operators)
    formulae[sensor_index] = spec
    if not os.path.exists(spec_file):
        with open(spec_file, "w"):
            pass
    with open(spec_file, "r+") as s:
        old_spec = s.read()
        if old_spec == repr(spec):
            return formulae
        s.seek(0)
        s.write(repr(spec))
        s.truncate()
    return formulae


def log_anomaly(
    trace, sensor_index, operators: list, tree=None, warmup2=False, sensor_type=None
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
            np.append(trace_np, anomaly_type).reshape(1, -1), batch_size=trace_np.size, binary=False, max_depth=config["TREE_CONFIG"]["MAX_DEPTH"], operators=operators
        )
        print("Tree built!")
    else:
        tree.update_tree(trace=np.append(trace_np, anomaly_type), batch_size=trace_np.size, binary=False, operators=operators)
        tree.print_tree()
    return True, tree

def apply_anomaly(data: np.ndarray, anomaly_indices: np.ndarray, anom_type: str) -> np.ndarray:
    if anom_type != "normal":
        print(data.shape)
        anomaly_size = data[:, anomaly_indices].std(axis=0)
        print(anom_type, anomaly_size)
        if anom_type == "small":
            anomaly_size /= 2
        data[:, anomaly_indices] += anomaly_size
    return data