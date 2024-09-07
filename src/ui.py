print("Loading UI features...")
import numpy as np
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from regressor import LargeWeightsRegressor
from preproc import preprocess_trace
import pandas as pd
from file_io import get_filename
from datetime import datetime, timedelta
from tree.new_formula import G

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
        while True:
            pressure_ids_given = input()
            temp_ids_given = input("Choose temperature sensor(s) to increase:\n")
            try:
                pressure_ids_list = pressure_ids_given.split(" ") if pressure_ids_given else []
                temp_ids_list = temp_ids_given.split(" ") if temp_ids_given else []
                ids_list = np.array(pressure_ids_list + temp_ids_list, dtype=int) - 1
                anom_type = "small" if response == "1" else "large"
                return anom_type, ids_list
            except:
                print("Invalid input. Please try again.")
                print("Choose pressure sensor(s) to increase:")
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


def print_trees(typed_anom_classifiers, parser):
    def display(sensor_index, sensor_type):
        tree = typed_anom_classifiers[sensor_type][sensor_index]
        if tree is None:
            print("No anomalies detected yet for this sensor.")
        else:
            tree.print_tree()

    while True:
        try:
            index_response = input("Enter sensor index: ")
            sensor_index = int(index_response) - 1
        except ValueError:
            print(f"Invalid sensor index: '{index_response}'")
            continue
        sensor_type = input(f"Enter sensor type ({'/'.join(parser.type)}): ")
        while sensor_type not in parser.type:
            sensor_type = input(f"Invalid sensor type: '{sensor_type}'")
        if not display(sensor_index=sensor_index, sensor_type=sensor_type):
            return
        continue_response = input("Show more graphs? (Y/n): ").lower()
        if continue_response not in ['yes', 'y']:
            return

def progress_bar(index, warmup_stage: int, warmup_times: list, bar_length=40):
    total_time = warmup_times[warmup_stage-1]
    progress = float(index) / float(total_time)
    block = int(bar_length * progress)
    text = (
        f"\rWarmup {warmup_stage}: Press Enter to continue... "
        f"[{index}/{total_time}] "
        f"[{'#' * block + '-' * (bar_length - block)}]"
    )
    sys.stdout.write(text)
    sys.stdout.flush()

def show_weights(parser) -> None:
    def display(sensor_index, sensor_type):
        filename = get_filename("weights", sensor_index, sensor_type=sensor_type)
        try:
            df = pd.read_csv(filename)
        except pd.errors.EmptyDataError:
            print("Model not fully trained yet.")
            return False
        indices = df.columns.astype(int).to_numpy()
        adjusted_indices = [i + 1 if i < sensor_index else i for i in indices]
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
        save_file = f"{config['GRAPH_DIR']}/{sensor_type.lower()}/sensor_{sensor_index+1}.png"
        plt.figure(figsize=(8, 5))
        plt.bar(adjusted_indices, weights, color=colours)
        plt.xlabel('Feature Index')
        plt.ylabel('Weights')
        plt.title('Model Weights')
        plt.grid(True)
        plt.xticks(adjusted_indices)
        plt.savefig(save_file)
        plt.show()
        return True
    while True:
        try:
            index_response = input("Enter sensor index: ")
            sensor_index = int(index_response) - 1
        except ValueError:
            print(f"Invalid sensor index: '{index_response}'")
            continue
        sensor_type = input(f"Enter sensor type ({'/'.join(parser.type)}): ")
        while sensor_type not in parser.type:
            sensor_type = input(f"Invalid sensor type: '{sensor_type}'")
        if not display(sensor_index=sensor_index, sensor_type=sensor_type.upper()):
            return
        continue_response = input("Show more graphs? (Y/n): ").lower()
        if continue_response not in ['yes', 'y']:
            return

def get_graph(sensor_type_index, parser, safe_trace_file=config["SAFE_TRACE_FILE"]):
    all_weights = []
    all_edges = []
    all_data = preprocess_trace(infile=safe_trace_file)
    indices = np.arange(parser.type_indices[sensor_type_index], parser.type_indices[sensor_type_index+1])
    relevant_data = all_data[:, indices]
    model = LargeWeightsRegressor(sensor_index=0)
    for sensor_index in range(len(indices)):
        model.set_sensor_index(sensor_index)
        X = np.delete(relevant_data, sensor_index, axis=1)
        y = relevant_data[:, sensor_index].astype(float)
        model.fit(X, y)
        weights = model.coef_
        edges = model.sensors_used
        all_weights.append(weights)
        all_edges.append(edges)
    return all_weights, all_edges

def draw_graph(edges, weights, bidirectional_only, sensor_type):
    save_file = f"{config['GRAPH_DIR']}/{sensor_type.lower()}/sensor_map.png"
    G = nx.DiGraph()
    for i, edge_array in enumerate(edges):
        for j, edge in enumerate(edge_array):
            if bidirectional_only:
                if i in edges[edge]:
                    weight_to_i = weights[edge][np.where(edges[edge] == i)[0][0]]
                    weight_to_edge = weights[i][j]
                    G.add_edge(f"S{edge+1}", f"S{i+1}", weight=np.round(weight_to_i, 3))
                    G.add_edge(f"S{i+1}", f"S{edge+1}", weight=np.round(weight_to_edge, 3))
            else:
                weight_to_edge = weights[i][j]
                G.add_edge(f"S{i+1}", f"S{edge+1}", weight=np.round(weight_to_edge, 3))
    for i in range(len(edges)):
        G.add_node(f"S{i+1}")
    pos = nx.spring_layout(G)
    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, font_size=10, font_color='black', ax=ax)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    ax.set_title(f"Graph of {sensor_type.capitalize()} Sensor Weights", fontsize=16, pad=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_file)
    plt.show()

def plot_graph(parser):
    bidirectional_only = config["BIDIRECTIONAL_ONLY"]
    for i, sensor_type in enumerate(parser.type):
        weights, edges = get_graph(sensor_type_index=i, parser=parser)
        draw_graph(edges=edges, weights=weights, bidirectional_only=bidirectional_only, sensor_type=sensor_type)

def print_anomaly_info(model, new_batch, formula):
    print("\nAnomaly detected!\n")
    data = preprocess_trace(new_batch=new_batch)
    X = np.delete(data, model.sensor_index, axis=1)
    y = data[:, model.sensor_index].astype(float)
    predictions = model.predict(X)
    print(f"Sensor {model.sensor_index+1} was calculated as:")
    for i, (weight, index) in enumerate(zip(model.coef_, model.indices_used)):
        start = "\t" if i == 0 else " \t+ "
        print(f"{start}Sensor {index+1} x {weight}")
    print(f"Predicted average was {predictions.mean()}")
    print(f"Actual average was {y.mean()}")
    print(f"STL formula was: {formula}")

def get_time_period(new_batch):
    first = new_batch[0] # ... ,284.969993591309,283.569993972778,01/01/2023,00:00:00
    second = new_batch[1] # ...,284.939993858337,283.589993476868,01/01/2023,00:15:00
    time_format = "%H:%M:%S"
    first_time = datetime.strptime(first[-9:-1], time_format)
    second_time = datetime.strptime(second[-9:-1], time_format)
    diff = second_time - first_time
    time_period = int(diff.total_seconds() / 60)
    return time_period

def get_and_display_anomaly_times(anomaly_indices: list, formula, new_batch: list, prev_backlog_size: int, end: int) -> None:
    time_period = get_time_period(new_batch)
    def get_anomaly_bounds(indices) -> list:
        if isinstance(formula, G):
            return ((0, len(new_batch)),)
        bounds = []
        N = len(indices)
        start_bound = None
        for i in range(N):
            this_value = indices[i]
            if i == 0 or indices[i-1] + 1 != this_value:
                start_bound = this_value - prev_backlog_size
            if i+1 == N or indices[i+1] - 1 != this_value:
                bounds.append((start_bound, this_value + end - prev_backlog_size - 1))
        return bounds
    print("Formula was:", formula)
    print(f"This means: {formula.human_readable(time_period)}.")
    print("This was not satisfied between the following times:")
    first_reading_values = new_batch[0].split(",")
    date = first_reading_values[-2]
    time = first_reading_values[-1].strip()
    datetime_str = f"{date} {time}"
    start_time = datetime.strptime(datetime_str, "%d/%m/%Y %H:%M:%S")
    bounds = get_anomaly_bounds(anomaly_indices)
    for interval in bounds:
        interval_start = (start_time + timedelta(minutes = (int(interval[0])) * time_period)).strftime("%d/%m/%Y %H:%M:%S")
        interval_end = (start_time + timedelta(minutes = (int(interval[1])) * time_period)).strftime("%d/%m/%Y %H:%M:%S")
        print(f"\t{interval_start} to {interval_end}")
    return bounds, start_time
    # times = [str((start_time + timedelta(minutes=i * frequency)).time()) for i in anomaly_indices]
    

def print_intro(types: list, warmup_times: tuple):
    formatted_types = f"{', '.join(types[:-1])} and {types[-1]}"
    print("=" * 65)
    print(f"Online {formatted_types} monitor".center(65))
    print("=" * 65)
    print("Instructions:")
    print("  - Enter : Read the next batch.")
    print("  - 'q'   : Quit the application.")
    print("  - 'a'   : Add a synthetic anomaly.")
    print("  - 'g'   : Display a graph showing connections between sensors.")
    print("  - 'w'   : Display the weights of the model for a particular sensor.")
    print("  - 'p'   : Display the STL formulae for each sensor.")
    print("=" * 65)
    print("\nNote:")
    print(f"  - There are two 'warmup' phases of length {' and '.join(warmup_times)},")
    print("    which must be completed before monitoring begins.")
    print("  - Synthetic anomalies cannot be added during the warmup phases.")
    print("=" * 65) 

if __name__ == "__main__":
    from tree.formula import Formula
    anomaly_indices = [1,2,3,5,6]
    formula = Formula.build_formula(0.1, "F", 6, "<=")
    new_batch = ['PDM23;03/01/2023;00:00:00;0.0389000015258789;Pressione a valle\n', 
                 'PDM24;03/01/2023;00:00:00;0.0362999992370605;Pressione a valle\n']
    get_and_display_anomaly_times(anomaly_indices=anomaly_indices, new_batch=new_batch, formula=formula)