print("Loading resources...")

from ui import read_user_input, print_trees, progress_bar, print_intro
from graphs import show_weights, plot_graph
from file_io import clear_files, get_new_batch, write_new_batch, end_anomaly, start_anomaly, get_filename
from model import get_residuals, update_spec, log_anomaly, new_batch_ok
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
    anomaly_statuses = []
    prev_safe_raw = {t: None for t in parser.type}
    prev_safe_res = {t: [] for t in parser.type}
    warmup_times = parser.w_1, parser.w_2
    progress_bar(index=0, warmup_stage=1, warmup_times=warmup_times)
    while True:
        prompt = " " if warmup1 or warmup2 else ">"
        response = read_user_input(prompt)
        if response == "q":
            print("Exiting monitor...")
            exit()
        if response == "p":
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
        new_batch = get_new_batch(
            batch_size=parser.batch,
            index=index,
            source_file=parser.infile,
        )
        index += 1
        if warmup1:
            progress_bar(warmup_stage=1, index=index, warmup_times=warmup_times)
            if index >= parser.w_1:
                warmup1 = False
                warmup2 = True
                print("\nWarmup 1 complete.")
            write_new_batch(new_batch=new_batch, outfile=safe_trace_file)
            continue
        if warmup2:
            progress_bar(warmup_stage=2, index = index - parser.w_1, warmup_times=warmup_times)
            if index >= parser.w_1 + parser.w_2:
                warmup2 = False
                print("\nWarmup complete.")
        test = preprocess_trace(new_batch=new_batch)
        train = preprocess_trace(infile=safe_trace_file)
        for i, sensor_type in enumerate(parser.type):
            prev_type = list(parser.type)[i-1]
            typed_formulae[prev_type] = copy.deepcopy(formulae)
            typed_anomaly_statuses[prev_type] = anomaly_statuses.copy()
            anomaly_statuses = typed_anomaly_statuses[sensor_type]
            formulae = typed_formulae[sensor_type]
            indices_used = np.arange(parser.type_indices[i], parser.type_indices[i+1])
            train_used = train[:, indices_used]
            test_used = test[:, indices_used]
            num_evaluations = len(indices_used)
            for sensor_index in range(num_evaluations):
                if sensor_index == len(prev_safe_res[sensor_type]):
                    prev_safe_res[sensor_type].append(None)
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
                        prev_safe_res[sensor_type][sensor_index] = None
                        if not anomaly_statuses[sensor_index]:
                            start_anomaly(new_batch, sensor_index + 1)
                        anomaly_statuses[sensor_index] = True
                        continue
                if sensor_index < len(anomaly_statuses) and anomaly_statuses[sensor_index]:
                    end_anomaly(new_batch, sensor_index + 1)
                    anomaly_statuses[sensor_index] = False
                residuals_file = get_filename("residuals", sensor_index=sensor_index, sensor_type=sensor_type)
                if warmup2:
                    with open(residuals_file, "a") as f:
                        f.write("\n" + new_trace)
                else:
                    prev_res = prev_safe_res[sensor_type][sensor_index]
                    if prev_res is not None:
                        with open(residuals_file, "a") as f:
                            f.write("\n" + prev_res)
                    prev_safe_res[sensor_type][sensor_index] = new_trace
                    formulae = update_spec(
                        formulae=formulae,
                        sensor_index=sensor_index,
                        operators=parser.stl,
                        sensor_type=sensor_type,
                    )
        all_statuses = list(typed_anomaly_statuses.values())
        all_statuses_concat = np.array(reduce(lambda x, y: x + y, all_statuses))
        if warmup2:
            write_new_batch(new_batch=new_batch, outfile=safe_trace_file)
        elif not np.any(all_statuses_concat):
            if prev_safe_raw[sensor_type] is not None:
                write_new_batch(new_batch=prev_safe_raw[sensor_type], outfile=safe_trace_file)
            prev_safe_raw[sensor_type] = new_batch
        else:
            prev_safe_raw[sensor_type] = None

def run_monitor(parser) -> None:
    clear_files(parser.type)
    w_1 = int(np.ceil(config["WARMUP_1_PROPORTION"] * parser.safe))
    w_2 = parser.safe - w_1
    warmup_times = str(w_1), str(w_2)
    print_intro(types=list(parser.type), warmup_times=warmup_times)
    parser.w_1 = w_1
    parser.w_2 = w_2
    monitor_loop(parser)