print("Loading formula synthesis functions...")
from tree.formula import Always, Formula
import numpy as np
import json
with open("config.json") as config_file:
    config = json.load(config_file)

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
            evaluation = formula.evaluate3(traces, labels=False)
            threshold = evaluation.mean(axis=1).min() if config["USE_MEAN"] else evaluation.min(axis=1).min()
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
