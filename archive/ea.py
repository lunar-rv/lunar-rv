import matplotlib.pyplot as plt
from new_formula import Formula
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from searching import hill_climbing_search, simulated_annealing_search, grid_search

contraction_fn = lambda r, b, max_size: r * np.exp((0.5 * b / max_size)-0.5)

def evaluate_formula(traces, F_end, G_avg_end):
    formula = Formula.build_formula(
        traces=traces,
        F_end=F_end,
        G_avg_end=G_avg_end,
    )
    rho = formula.evaluate(traces=traces, labels=False)
    r = -rho.ptp()
    b = formula.f.end
    max_size = 96
    score = contraction_fn(r, b, max_size)
    return score

def create_heatmap(scores):
    mu = np.mean(scores)
    sigma = np.std(scores)
    scores_scaled = (scores - mu) / sigma
    # Reshape the scaled array back to the original shape
    plt.title("Heatmap of Formula Scores")
    plt.ylabel("F interval")
    plt.xlabel("G_avg interval")
    plt.imshow(scores_scaled, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=scores_scaled.max())
    plt.show()

def main():
    # traces = np.genfromtxt("inputs/pressure_residuals.csv", delimiter=",", dtype=float)
    all_traces = np.load("numpy/pressure_residuals_all.npy")
    batch_size = 96
    all_scores = np.empty((27))
    for sensor_index in range(27):
        traces = all_traces[sensor_index]
        print("Sensor", sensor_index)
        scores = np.empty((batch_size-1, batch_size-1))
        # for i in range(1, batch_size):
        #     for j in range(1, batch_size):
        #         scores[i-1, j-1] = evaluate_formula(traces, i, j)
        best_x, best_y, score = grid_search(traces, batch_size, evaluate_formula)
        # best_indices = np.unravel_index(np.argmax(scores), scores.shape)
        print(f"Best indices: ({best_x}, {best_y})")
        print("Best score:", score)
        all_scores[sensor_index] = score
    print(all_scores.tolist())
    # np.save("scores.npy", all_scores)

if __name__ == "__main__":
    main()
    
    # create_heatmap(np.load("scores.npy"))