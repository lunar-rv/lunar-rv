print("Loading graph features...")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import json
from file_io import get_filename

with open("config.json", "r") as config_file:
    config = json.load(config_file)

logging.getLogger('matplotlib').setLevel(logging.WARNING)

def plot_traces(neg_infile, pos_infile, outfile):
    neg_df = pd.read_csv(neg_infile)
    pos_df = pd.read_csv(pos_infile)
    negatives = neg_df.to_numpy().flatten()
    positives = pos_df.to_numpy().flatten()
    plt.plot(negatives[:96], label="Negatives", color="blue")
    plt.plot(positives[:96], label="Positives", color="red")
    plt.xlabel("Time")
    plt.ylabel("Magnitude of residual for day 1")
    plt.legend()
    plt.savefig(outfile)
    plt.close()


def plot_array(trace: np.ndarray, sensor_index: int, batch_start_time: datetime, keyword: str, sensor_type: str, backlog_size: int = 0, formula=None, bounds: np.ndarray = np.array([]), time_period=-1, preds=None):
    trace_start_time = batch_start_time - timedelta(minutes=backlog_size * time_period)
    trace_end_time = trace_start_time + timedelta(minutes=(len(trace) - 1) * time_period)
    int_ticks = np.linspace(0, len(trace)-1, min(len(trace), 9))
    dt_ticks = [trace_start_time + timedelta(minutes=int(tick) * time_period) for tick in int_ticks]
    plt.plot(trace, label=f"{keyword}", color='blue')
    if preds is not None:
        plt.plot(preds, label="Predictions", color='green')
    for start, end in bounds:
        plt.axvspan(start, end, color='orange', alpha=0.3, label="Anomaly" if start == bounds[0][0] else "")
    plt.xlabel("Time")
    plt.xticks(int_ticks, [dt.strftime("%H:%M") for dt in dt_ticks])
    plt.ylabel(f"Sensor {sensor_index+1} {keyword}")
    
    if formula is not None:
        plt.axhline(y=formula.boundary, color="red", linestyle="--", label="Formula boundary")
    
    if trace_start_time.date == trace_end_time.date:
        date_str = f"on {trace_start_time.strftime('%Y/%m/%d')}"
    else:
        date_str = f"from {trace_start_time.strftime('%Y/%m/%d')} to {trace_end_time.strftime('%Y/%m/%d')}"
    
    title = f"{keyword} for {sensor_type.capitalize()} Sensor {sensor_index+1} {date_str}"
    
    if formula is not None:
        subtext = f"Formula: {formula}"
        if formula.boundary is not None and not formula.name == "G":
            subtext += f"\nAnomaly Bounds: {bounds.tolist() if bounds is not None else None}"
        plt.text(0.5, -0.15, subtext, fontsize=10, verticalalignment='top', 
                    horizontalalignment='center', transform=plt.gca().transAxes,
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    plt.title(title)
    plt.legend()
    plt.subplots_adjust(bottom=0.2)
    plt.show()

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
    from preproc import preprocess_trace
    from regressor import LargeWeightsRegressor
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