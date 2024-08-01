from tree.formula import Always, Formula
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib import pyplot as plt


def plot(arr, boundary: float = None, title=""):
    plt.figure(figsize=(10, 6))
    plt.plot(arr)
    plt.xlabel("Trace")
    plt.ylabel("Robustness")
    if boundary:
        plt.axhline(boundary, color="red", label=f"Mean prediction error <= {boundary}")
    plt.title(title)
    plt.legend()
    plt.show()


def classify_traces(traces, phi, plot_rob=False):
    split_traces = traces.reshape((traces.shape[0], phi.end, -1))
    rob = (
        np.max(split_traces, axis=1)
        if isinstance(phi, Always)
        else np.min(split_traces, axis=1)
    )
    mean_rob = rob.mean(axis=1)
    if plot_rob:
        plot(mean_rob, boundary=phi.boundary)
    classifications = np.where(mean_rob <= phi.boundary, True, False)
    return classifications


def default_best():
    return {"interval": -1, "value": np.inf, "op": "", "threshold": -1}


def positive_synth(traces, best=default_best(), operators="FG_", invariance=False, use_mean=True):
    if invariance:
        max_threshold = traces.mean(axis=2).max() if use_mean else traces.max()
        boundary = 1 if use_mean else 96
        return Formula.build_formula(max_threshold, "G", boundary)
    if "F" in operators:
        ev_rob = np.min(traces, axis=1)  # Pick the best value
        ev_mean_rob = ev_rob.mean(axis=1)
        ev_value = ev_mean_rob.ptp()
    else:
        ev_mean_rob, ev_value = np.inf, np.inf
    if "G" in operators:
        alw_rob = np.max(traces, axis=1)  # Pick the worst value
        alw_mean_rob = alw_rob.mean(axis=1)
        alw_value = alw_mean_rob.ptp()
    else:
        alw_mean_rob, alw_value = np.inf, np.inf
    if ev_value > alw_value:
        value, op, threshold = ev_value, "F", ev_mean_rob.max()
    else:
        value, op, threshold = alw_value, "G", alw_mean_rob.max()
    value, op, threshold = ev_value, "F", ev_mean_rob.max()
    if value < best["value"]:
        best["value"] = value
        best["interval"] = traces.shape[1]
        best["op"] = op
        best["threshold"] = threshold
    if traces.shape[2] % 2 != 0:
        return Formula.build_formula(best["threshold"], best["op"], best["interval"])
    new_shape = (-1, traces.shape[1] * 2, traces.shape[2] // 2)
    return positive_synth(traces.reshape(new_shape), best, operators=operators, invariance=invariance)


def main():
    # predict(anomaly_size=0.0003)
    predictions_file = "csv/predictions.csv"
    traces = np.genfromtxt(predictions_file, delimiter=",", dtype=float)
    testing = False
    filetype = "test" if testing else "val"
    neg_infile = f"csv/negative_{filetype}.csv"
    pos_infile = f"csv/positive_{filetype}.csv"
    negatives = np.genfromtxt(neg_infile, delimiter=",", dtype=float)  # no anomalies
    positives = np.genfromtxt(pos_infile, delimiter=",", dtype=float)  # has anomalies
    num_sensors = 27
    neg_classifications = []
    pos_classifications = []
    traces = traces.reshape(num_sensors, traces.shape[1], -1)
    negatives = negatives.reshape(num_sensors, negatives.shape[1], -1)
    positives = positives.reshape(num_sensors, positives.shape[1], -1)
    for sensor_index in range(num_sensors):
        if sensor_index != 0:
            continue
        sensor_traces = traces[sensor_index]
        formula = positive_synth(sensor_traces[:, np.newaxis, :], best=default_best())
        print(f"Sensor {sensor_index+1} formula: {formula}")
        neg_classifications += classify_traces(negatives[sensor_index], formula).tolist()
        pos_classifications += classify_traces(positives[sensor_index], formula).tolist()
    ground_truth_neg = np.full_like(neg_classifications, False)
    ground_truth_pos = np.full_like(pos_classifications, True)
    ground_truth = np.concatenate([ground_truth_neg, ground_truth_pos])
    predictions = ~np.concatenate([neg_classifications, pos_classifications])
    print(f"Accuracy: {accuracy_score(ground_truth, predictions)}")
    print(f"Precision: {precision_score(ground_truth, predictions, zero_division=0)}")
    print(f"Recall: {recall_score(ground_truth, predictions, zero_division=0)}")
    print(f"F1: {f1_score(ground_truth, predictions, zero_division=0)}")
    # print("=" * 50)
    # print("Non-anomalous predictions:", ~np.array(neg_classifications))
    # print("=" * 50)
    # print("Anomalous predictions:", ~np.array(pos_classifications))
    # print("=" * 50)
    # print("Ground truth:", ground_truth)


if __name__ == "__main__":
    traces = np.array([
        [1,2,1,1],
        [3,4,2,1],
    ])
    print(positive_synth(traces[:, np.newaxis, :], invariance=True))
    #main()
