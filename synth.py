print("Loading formula synthesis functions...")
from searching import hill_climbing_search
from tree.new_formula import FormulaFactory
import numpy as np
import json
with open("config.json") as config_file:
    config = json.load(config_file)

# FIX THE CLOSENESS FUNCTION.
contraction_fn = lambda r, b, max_size: r * np.exp((0.5 * b / max_size)-0.5)

def evaluate_formula(traces, F_end, G_avg_end):
    formula = FormulaFactory.build_tightest_formula(
        traces=traces,
        F_end=F_end,
        G_avg_end=G_avg_end,
    )
    rho = formula.evaluate(traces=traces, labels=False).min(axis=1)
    r = -rho.ptp()
    b = formula.f.end
    max_size = config["BATCH_SIZE"]
    score = contraction_fn(r, b, max_size)
    return score

def positive_synth(traces, prev_formula=None, reading_type="PRESSURE"):
    best_formula = None
    batch_size = traces.size // traces.shape[0]
    best_x, best_y, _ = hill_climbing_search(traces, batch_size, evaluate_formula)
    best_formula = FormulaFactory.build_tightest_formula(
        traces=traces,
        F_end=best_x,
        G_avg_end=best_y,
        reading_type=reading_type
    )
    if prev_formula:
        lrv = prev_formula.last_raw_values
        lr = prev_formula.last_residuals
        if prev_formula.max_length > best_formula.max_length:
            lrv = lrv[1-best_formula.max_length:]
            lr = lr[1-best_formula.max_length:]
        elif prev_formula.max_length < best_formula.max_length:
            padding_length = best_formula.max_length - prev_formula.max_length
            lrv = np.pad(lrv, ((0, 0), (padding_length, 0)), mode='constant', constant_values=np.nan)
            lr = np.pad(lr, ((0, 0), (padding_length, 0)), mode='constant', constant_values=np.nan)
        best_formula.last_raw_values = lrv
        best_formula.last_residuals = lr
    return best_formula

def main():
    import telex.synth as tx
    import time
    from formula import Formula
    import matplotlib.pyplot as plt
    traces = np.genfromtxt("inputs/pressure_residuals.csv", delimiter=",", dtype=float)
    traces = traces[:2, :]
    traces *= 1000
    # print(traces.shape)
    formula1 = positive_synth(traces)
    formula2 = Formula.build_formula(0.01247759930412972, "F", 3, "<=")
    eval1 = formula1.evaluate3(traces, labels=False).flatten()
    eval2 = formula2.evaluate3(traces, labels=False).flatten()
    plt.plot(eval1, label="New positive synthesis")
    plt.plot(eval2, label="TeLEX")
    plt.legend()
    plt.title("Robustness of synthesised formulae on pressure residual traces over 2 days")
    plt.xlabel("Time")
    plt.ylabel("Robustness")
    plt.show()
    exit()

if __name__ == "__main__":
    main()