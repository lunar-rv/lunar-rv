print("Loading resources...")

from ui import read_user_input, read_anomaly_indices, print_trees, progress_bar, show_weights, plot_graph, print_intro
from file_io import clear_files, get_new_batch, write_new_batch, end_anomaly, start_anomaly, get_filename
from model import get_residuals, update_spec, log_anomaly, new_batch_ok, apply_anomaly
from functools import reduce
from preproc import preprocess_trace
import numpy as np
import json
import copy

with open("config.json", 'r') as config_file:
    config = json.load(config_file)

def monitor_loop(parser) -> None:
    index = 0
    warmup1 = True
    warmup2 = False
    safe_trace_file = config["SAFE_TRACE_FILE"]
    trees = {
        t: np.full((parser.type_indices[i+1] - parser.type_indices[i]), None, dtype=object) 
        for i, t in enumerate(parser.type)
    }
    typed_formulae = {t: [] for t in parser.type}
    typed_anomaly_statuses = {t: [] for t in parser.type}
    formulae = []
    bin_classifiers = []
    anomaly_statuses = []
    warmup_1_time = int(np.ceil(config["WARMUP_1_PROPORTION"] * parser.safe))
    warmup_2_time = parser.safe - warmup_1_time
    warmup_times = (warmup_1_time, warmup_2_time)
    progress_bar(index=0, warmup_stage=1, warmup_times=warmup_times)
    while True:
        prompt = " " if warmup1 or warmup2 else ">"
        response = read_user_input(prompt)
        if response == "q":
            print("Exiting monitor...")
            exit()
        if response == "p":
            print(trees.items())
            print_trees(trees, parser=parser)
            continue
        if response == "w":
            if warmup1:
                print("Connections not yet learnt.")
                continue
            show_weights(parser=parser)
            continue
        if response == "g":
            plot_graph(parser=parser)
            continue
        anomaly_info = read_anomaly_indices() if response == "a" and not (warmup1 or warmup2) else ("normal", np.array([]))
        anom_type, anomaly_indices = anomaly_info
        new_batch = get_new_batch(
            batch_size=parser.batch,
            index=index,
            source_file=parser.infile,
        )
        index += 1
        if warmup1:
            progress_bar(warmup_stage=1, index=index, warmup_times=warmup_times)
            if index >= warmup_1_time:
                warmup1 = False
                warmup2 = True
                print("\nWarmup 1 complete.")
            write_new_batch(new_batch=new_batch, outfile=safe_trace_file)
            continue
        if warmup2:
            progress_bar(warmup_stage=2, index = index - warmup_1_time, warmup_times=warmup_times)
            if index >= warmup_1_time + warmup_2_time:
                warmup2 = False
                print("\nWarmup complete.")
        test = preprocess_trace(new_batch=new_batch)
        train = preprocess_trace(infile=safe_trace_file)
        for i, sensor_type in enumerate(parser.type):
            prev_type = list(parser.type)[i-1]
            # typed_bin_classifiers[prev_type] = copy.deepcopy(bin_classifiers)
            typed_formulae[prev_type] = copy.deepcopy(formulae)
            typed_anomaly_statuses[prev_type] = anomaly_statuses.copy()
            anomaly_statuses = typed_anomaly_statuses[sensor_type]
            # bin_classifiers = typed_bin_classifiers[sensor_type]
            formulae = typed_formulae[sensor_type]
            indices_used = np.arange(parser.type_indices[i], parser.type_indices[i+1])
            train_used = train[:, indices_used]
            test = apply_anomaly(data=test, anomaly_indices=anomaly_indices, anom_type=anom_type)
            test_used = test[:, indices_used]
            num_evaluations = len(indices_used)
            for sensor_index in range(num_evaluations):
                if not warmup2:
                    print(f"{sensor_type.upper()} SENSOR {sensor_index + 1}")
                elif len(anomaly_statuses) <= sensor_index:
                    anomaly_statuses.append(False)
                    formulae.append(None)
                start = max(len(train_used) - config["WINDOW_SIZE"] * parser.batch, 0)
                residuals = get_residuals(
                    train=train_used[start:],
                    test=test_used,
                    sensor_index=sensor_index,
                    sensor_type=sensor_type,
                )
                new_trace = ",".join(np.abs(residuals).astype(str))
                formula = formulae[sensor_index]
                if not new_batch_ok(residuals=residuals, start_index=parser.type_indices[i], formula=formula, new_batch=new_batch, sensor_index=sensor_index, sensor_type=sensor_type):
                    confirmation, trees[sensor_type][sensor_index] = log_anomaly(
                        new_trace, sensor_index, operators=parser.stl, tree=trees[sensor_type][sensor_index], warmup2=warmup2, sensor_type=sensor_type
                    )
                    if confirmation:
                        if not anomaly_statuses[sensor_index]:
                            start_anomaly(new_batch, sensor_index + 1)
                        anomaly_statuses[sensor_index] = True
                        formulae, bin_classifiers = update_spec(
                            formulae=formulae,
                            operators=parser.stl,
                            sensor_index=sensor_index,
                            bin_classifier=bin_classifiers,
                            new_trace=np.abs(residuals),
                            new_label="Anomaly",
                            sensor_type=sensor_type,
                        )
                        continue
                if sensor_index < len(anomaly_statuses) and anomaly_statuses[sensor_index]:
                    end_anomaly(new_batch, sensor_index + 1)
                    anomaly_statuses[sensor_index] = False
                residuals_file = get_filename("residuals", sensor_index=sensor_index, sensor_type=sensor_type)
                with open(residuals_file, "a") as f:
                    f.write("\n" + new_trace)
                if not warmup2:
                    formulae, bin_classifiers = update_spec(
                        formulae=formulae,
                        sensor_index=sensor_index,
                        bin_classifier=bin_classifiers,
                        operators=parser.stl,
                        new_trace=np.abs(residuals),
                        new_label="Safe",
                        sensor_type=sensor_type,
                    )
        all_statuses = list(typed_anomaly_statuses.values())
        all_statuses_concat = np.array(reduce(lambda x, y: x + y, all_statuses))
        if warmup2 or not np.any(all_statuses_concat):
            write_new_batch(new_batch=new_batch, outfile=safe_trace_file)


def run_monitor(parser) -> None:
    clear_files(parser.type)
    warmup_times = (str(int(np.ceil(config["WARMUP_1_PROPORTION"] * parser.safe))), 
                    str(parser.safe - int(np.ceil(config["WARMUP_1_PROPORTION"] * parser.safe)))
    )
    print_intro(types=list(parser.type), warmup_times=warmup_times)
    monitor_loop(parser)

if __name__ == "__main__":
    from parser import Parser
    parser = Parser()
    parser.infile = "inputs/reversed.csv"
    run_monitor(parser)
