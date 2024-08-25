import numpy as np
from sklearn.ensemble import RandomForestRegressor
from ui import plot_graph
import networkx as nx
import matplotlib.pyplot as plt
from regressor import LargeWeightsRegressor
from sklearn.linear_model import LinearRegression, Lasso


def get_graph():
    all_edges = []
    all_weights = []
    for sensor_index in range(27):
        print("Sensor index:", sensor_index+1)
        edges = []
        weights = []
        data = np.genfromtxt("inputs/preprocessed.csv", delimiter=",", dtype=float)
        np.set_printoptions(suppress=True)
        X = np.delete(data[:, :27], sensor_index, axis=1)
        y = data[:, sensor_index]
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        feature_importance_rf = rf.feature_importances_
        feature_indices = np.where(feature_importance_rf > 0.07)[0]
        for index in feature_indices:
            edges.append(index)
            weights.append(feature_importance_rf[index])
        all_edges.append(edges)
        all_weights.append(weights)
    for i, edges in enumerate(all_edges):
        for j, edge in enumerate(edges):
            if edge >= i:
                all_edges[i][j] += 1
    return all_edges, all_weights

def plot_graph(edges, weights, name="pressures.png"):
    G = nx.DiGraph()
    for i, edge_array in enumerate(edges):
        for j, edge in enumerate(edge_array):
            try:
                weight_to_edge = weights[i][j]
                G.add_edge(f"S{i+1}", f"S{edge+1}", weight=np.round(weight_to_edge, 3))
            except IndexError:
                print(f"IndexError: Could not find weight for edge {i} -> {edge}")
    for i in range(len(edges)):
        G.add_node(f"S{i+1}")
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='red', node_size=500, font_size=10, font_color='white')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    
    plt.title("Bidirectional Weighted Graph")
    plt.savefig(name)
    plt.show()

def get_lr_graph(pressures=True):
    all_weights = []
    all_edges = []
    all_data = np.genfromtxt("inputs/preprocessed.csv", delimiter=",", dtype=float)
    num_sensors = all_data.shape[1] // 2
    indices = np.arange(num_sensors) if pressures else np.arange(num_sensors, 2*num_sensors)
    relevant_data = all_data[:, indices]
    for sensor_index in range(num_sensors):
        print("Sensor index:", sensor_index+1)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        X = np.delete(relevant_data, sensor_index, axis=1)
        y = relevant_data[:, sensor_index].astype(float)
        rf.fit(X, y)
        weights = rf.feature_importances_
        print(weights.tolist())
        edges = np.arange(26)
        all_weights.append(weights)
        all_edges.append(edges)
    for i, edges in enumerate(all_edges):
        for j, edge in enumerate(edges):
            if edge >= i:
                all_edges[i][j] += 1
    return all_weights, all_edges



def compute_symmetry_score(weights, edges):
    num_sensors = len(weights)
    symmetry_scores = np.zeros((num_sensors, num_sensors))
    for i in range(num_sensors):
        for j in range(len(edges[i])):
            a_to_b_weight = weights[i][j]
            b = edges[i][j]
            b_to_a_weight = weights[b][np.where(edges[b] == i)[0][0]]
            symmetry_scores[i][b] = min(a_to_b_weight, b_to_a_weight)
    return symmetry_scores

# Calculate the symmetry scores
def main():
    weights, edges = get_lr_graph()
    symmetry_scores = compute_symmetry_score(weights, edges)

    # Display the symmetry scores
    print("Symmetry Scores (Bidirectional Strength):")
    print(symmetry_scores.mean())

if __name__ == "__main__":
    main()