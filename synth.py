print("Loading formula synthesis functions...")
from tree.formula import Always, Formula
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib import pyplot as plt
import json
with open("config.json") as config_file:
    config = json.load(config_file)

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

# FIX THE CLOSENESS FUNCTION.
def weighted_closeness(arr: np.ndarray, interval: int, trace_length=-1) -> float:
    gamma = config["GAMMA"]
    contraction_fn = lambda r, size: r * (1 + np.exp((size - 0.5) * gamma))
    score = -arr.ptp() # Aim to maximise score
    proportion = interval / trace_length # Log may not be helpful actually
    result = contraction_fn(score, proportion)
    return result
    
def positive_synth(traces, operator="F", prev_formula=None):
    trace_length = traces.size // traces.shape[0]
    best_score = -np.inf
    best_formula = None
    if operator == "F":
        for i in range(1, trace_length):
            formula = Formula.build_formula(0, "F", i, "<=")
            evaluation = formula.evaluate3(traces, labels=False).min(axis=1)
            threshold = evaluation.min() if config["USE_MIN"] else evaluation.mean()
            score = weighted_closeness(evaluation, interval=i, trace_length=trace_length)
            if score > best_score:
                best_score = score
                best_formula = Formula.build_formula(-threshold, "F", i, "<=")
        if prev_formula:
            lrv = prev_formula.last_raw_values
            lr = prev_formula.last_residuals
            if prev_formula.end > best_formula.end:
                lrv = lrv[1-best_formula.end:]
                lr = lr[1-best_formula.end:]
            elif prev_formula.end < best_formula.end:
                padding_length = best_formula.end - prev_formula.end
                lrv = np.pad(lrv, ((0, 0), (padding_length, 0)), mode='constant', constant_values=np.nan)
                lr = np.pad(lr, ((0, 0), (padding_length, 0)), mode='constant', constant_values=np.nan)
            best_formula.last_raw_values = lrv
            best_formula.last_residuals = lr
    elif operator == "G":
        best_formula = Formula.build_formula(traces.max(), "G", "<=")
    return best_formula
    


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
        formula = positive_synth(sensor_traces[:, :, np.newaxis])
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
    traces = np.genfromtxt("outputs/residuals/pressure/sensor_1_residuals.csv", dtype=float, delimiter=",")
    formula = positive_synth(traces, operator="F")
    print(formula)
    #main()
