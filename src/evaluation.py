import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from tree.formula import Formula
from synth import positive_synth
import json
with open("config.json") as config_file:
    config = json.load(config_file)

def X_Y_split(data: np.ndarray, i: int, axis=1):
    index = [slice(None)] * data.ndim
    index[axis] = i
    Y = data[tuple(index)].astype(float)
    X = np.delete(data, i, axis=axis)
    return X, Y

def cut(data, dividend):
    return data[data.shape[0] % dividend:]

def print_metrics(FP, TN, FN, TP):
    print("=========================================")
    print("Evaluation on normal data:")
    print("=========================================")
    print("False Positives:", FP)
    print("True Negatives:", TN)
    print("=========================================")
    print("Evaluation on anomalous data:")
    print("=========================================")
    print("False Negatives:", FN)
    print("True Positives:", TP)
    print("=========================================")
    print("Recall:", TP / (TP + FN))
    print("Precision:", TP / (TP + FP))
    print("Accuracy:", (TP + TN) / (TP + TN + FP + FN))

def get_rob(formula, residuals, batch_size):
    residuals = cut(residuals, batch_size)
    residuals = residuals.reshape(-1, batch_size)
    evaluations = []
    for batch in residuals:
        e = formula.evaluate_single(batch, raw_values=np.array([""]), labels=False)
        if e.shape[0] != batch_size: # first batch
            e = np.hstack((np.zeros(batch_size - e.shape[0]), e))
        evaluations.append(e)
    evaluations = np.array(evaluations)
    if config["USE_MEAN"]:
        rob = evaluations.mean(axis=1) 
    else:
        rob = evaluations.min(axis=1)
    return rob

def evaluate_pressures(infile="inputs/pressures.csv"):
    pressures = np.genfromtxt(infile, delimiter=",", dtype=float)
    train_lr, other = train_test_split(pressures, test_size=0.9, random_state=42)
    train_stl, test = train_test_split(other, test_size=0.9, random_state=42)
    print(train_lr.shape, train_stl.shape, test.shape)
    batch_size = 48
    for sensor_index in range(1): # 27
        X_train_lr, Y_train_lr = X_Y_split(train_lr, sensor_index)
        X_train_stl, Y_train_stl = X_Y_split(train_stl, sensor_index)
        X_test, Y_test = X_Y_split(test, sensor_index)
        model = LinearRegression()
        model.fit(X_train_lr, Y_train_lr)
        stl_train_predictions = model.predict(X_train_stl)
        residuals = np.abs(stl_train_predictions - Y_train_stl) * 1000
        residuals = cut(residuals, batch_size)
        residuals = residuals.reshape(-1, batch_size)
        true_positives = []
        true_negatives = []
        for end in range(2, batch_size):
            print("Evaluating for interval length", end)
            formula = Formula.build_formula(0, "F", end, "<=")
            train_evaluations = formula.evaluate3(residuals, labels=False)
            threshold = train_evaluations.mean(axis=1).min() if config["USE_MEAN"] else train_evaluations.min(axis=1).min()
            formula = Formula.build_formula(-threshold, "F", end, "<=")
            print("Formula:", formula)
            test_predictions = model.predict(X_test)
            test_residuals = np.abs(test_predictions - Y_test) * 1000
            rob = get_rob(formula, test_residuals, batch_size)
            TN = len(rob[rob >= 0])
            Y_test_anom = Y_test + 0.0002
            test_residuals_anom = np.abs(test_predictions - Y_test_anom) * 1000
            rob_anom = get_rob(formula, test_residuals_anom, batch_size)
            # FN = len(rob_anom[rob_anom >= 0])
            TP = len(rob_anom[rob_anom < 0])
            true_negatives.append(TN)
            true_positives.append(TP)
        return true_negatives, true_positives
        # threshold = evaluation.min() ## if config["USE_MIN"] else evaluation.mean()


def plot_evaluations(TN, TP):
    import matplotlib.pyplot as plt
    plt.plot(TN, label="True Negatives")
    plt.plot(TP, label="True Positives")
    plt.legend()
    plt.show()

def main():
    tn, tp = evaluate_pressures()
    plot_evaluations(tn, tp)

if __name__ == "__main__":
    main()