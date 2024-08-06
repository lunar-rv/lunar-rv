import numpy as np
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from regressor import LargeWeightsRegressor
from preproc import preprocess
import pandas as pd
from file_io import get_filename

with open('config.json', 'r') as file:
    config = json.load(file)

def read_user_input(prompt=">") -> str:
    response = input(prompt)
    if not response:
        return "c"
    if response.lower() in "apqwg":
        return response.lower()
    return read_user_input(prompt=prompt)

def read_anomaly_indices() -> tuple:
    print("Choose anomaly type:")
    print("Press 1 for a small increase to individual sensors")
    print("Press 2 for a big increase to individual sensors")
    print("Press 3 for a network-wide increase")
    print("Or press 4 to enter a normal batch:", end=" ")
    response = input()
    while response not in "1234":
        response = input()
    if response in "12":
        print("Choose pressure sensor(s) to increase:")
        print("\t- Sensor IDs should be separated with spaces, e.g. 1 2 3")
        print("\t- IDs are expected to start from 1 ")
        pressure_ids_given = input()
        temp_ids_given = input("Choose temperature sensor(s) to increase:\n")
        ids_given = pressure_ids_given + " " + temp_ids_given
        anom_type = "small" if response == "1" else "large"
        return anom_type, np.array(ids_given.split(" "), dtype=int) - 1
    if response == "3":
        print("Choose anomaly size (small/large):")
        size_response = input()
        while size_response.lower() not in ["small", "large"]:
            print(f"Invalid response: '{response.lower()}'")
            size_response = input()
        return size_response.lower(), np.arange(54)
    if response == "4":
        return "normal", None

def print_score(tree):
    print("=" * 50)
    print("STL FORMULA WAS:")
    print(tree.to_stl())
    correct, total = tree.count_correct(head=True)
    print(f"Accuracy: {correct}/{total}")


def print_trees(bin_classifier, anom_classifier):
    if bin_classifier:
        print("ANOMALY/SAFE tree:")
        bin_classifier.print_tree()
        print_score(bin_classifier)
    if anom_classifier:
        print("ANOMALY TYPE CLASSIFICATION tree")
        anom_classifier.print_tree()
        print_score(anom_classifier)

def progress_bar(index, warmup_stage: int, bar_length=40):
    progress = float(index) / float(config["WARMUP_TIME"])
    block = int(round(bar_length * progress))
    text = f"\rWarmup {warmup_stage}: Press Enter to continue... [{index}/{config['WARMUP_TIME']}] [{'#' * block + '-' * (bar_length - block)}]"
    sys.stdout.write(text)
    sys.stdout.flush()

def show_weights(sensor_index, sensor_type) -> None:
    filename = get_filename("weights", sensor_index, sensor_type=sensor_type)
    df = pd.read_csv(filename)
    indices = df.columns.astype(int).to_numpy()
    weights = df.iloc[0].to_numpy()
    def weight_to_color(weight, max_positive_weight, max_negative_weight):
        if weight >= 0:
            normalized_weight = weight / max_positive_weight
            return plt.cm.Reds(normalized_weight)
        else:
            normalized_weight = -weight / max_negative_weight
            return plt.cm.Blues(normalized_weight)
    max_positive_weight = max(weights.max(), 1e-10)
    max_negative_weight = max(-weights.min(), 1e-10)
    colours = [weight_to_color(w, max_positive_weight, max_negative_weight) for w in weights]
    plt.figure(figsize=(8, 5))
    plt.bar(indices, weights, color=colours)
    plt.xlabel('Feature Index')
    plt.ylabel('Weights')
    plt.title('Model Weights')
    plt.grid(True)
    plt.xticks(indices)
    plt.savefig(config["WEIGHTS_GRAPH_FILE"])
    plt.show()

def get_graph(safe_trace_file=config["SAFE_TRACE_FILE"], pressures=True):
    all_weights = []
    all_edges = []
    all_data = preprocess(safe_trace_file)
    num_sensors = all_data.shape[1] // 2
    indices = np.arange(num_sensors) if pressures else np.arange(num_sensors, 2*num_sensors)
    relevant_data = all_data[:, indices]
    model = LargeWeightsRegressor(sensor_index=0)
    for sensor_index in range(num_sensors):
        model.set_sensor_index(sensor_index)
        X = np.delete(relevant_data, sensor_index, axis=1)
        y = relevant_data[:, sensor_index].astype(float)
        model.fit(X, y)
        weights = model.coef_
        edges = model.sensors_used
        all_weights.append(weights)
        all_edges.append(edges)
    return all_weights, all_edges

def plot_graph(name="pressures.png"):
    weights, edges = get_graph()
    G = nx.DiGraph()
    for i, edge_array in enumerate(edges):
        for j, edge in enumerate(edge_array):
            if i in edges[edge]:
                weight_to_i = weights[edge][np.where(edges[edge] == i)[0][0]]
                weight_to_edge = weights[i][j]
                G.add_edge(f"S{edge+1}", f"S{i+1}", weight=np.round(weight_to_i, 3))
                G.add_edge(f"S{i+1}", f"S{edge+1}", weight=np.round(weight_to_edge, 3))
    for i in range(len(edges)):
        G.add_node(f"S{i+1}")
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='red', node_size=500, font_size=10, font_color='white')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title("Bidirectional Weighted Graph")
    plt.savefig("temperatures.png")
    plt.show()

def print_anomaly_info(model, new_batch, formula):
    print("\nAnomaly detected!\n")
    data = preprocess("".join(new_batch), csv=False)
    X = np.delete(data, model.sensor_index, axis=1)
    y = data[:, model.sensor_index].astype(float)
    predictions = model.predict(X)
    print(f"Sensor {model.sensor_index+1} was calculated as:")
    for i, (weight, index) in enumerate(zip(model.coef_, model.indices_used)):
        start = "\t" if i == 0 else " \t+ "
        print(f"{start}Sensor {index+1} x {weight}")
    print(f"Predicted average was {predictions.mean() * 1000}")
    print(f"Actual average was {y.mean() * 1000}")
    print(f"STL formula was: {formula}")

def print_intro():
    print("=" * 65)
    print("Welcome to the Online Gas Network Monitor".center(65))
    print("=" * 65)
    print("Instructions:")
    print("  - Enter : Read the next batch.")
    print("  - 'q'   : Quit the application.")
    print("  - 'a'   : Add a synthetic anomaly.")
    print("  - 'g'   : Display a graph showing connections between sensors.")
    print("=" * 65)
    print("\nNote:")
    print(f"  - There are two 'warmup' phases of length {config['WARMUP_TIME']} each,")
    print("    which must be completed before monitoring begins.")
    print("  - Synthetic anomalies cannot be added during the warmup phases.")
    print("=" * 65)
