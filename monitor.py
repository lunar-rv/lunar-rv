print("Loading resources...")

from ui import read_user_input, read_anomaly_indices, print_trees, progress_bar, show_weights, plot_graph, print_intro
from file_io import clear_files, write_header, get_new_batch, write_new_batch, end_anomaly, start_anomaly, get_filename
from model import get_residuals, update_spec, log_anomaly, new_batch_ok
from preproc import preprocess
import numpy as np
import json



with open("config.json", 'r') as config_file:
    config = json.load(config_file)

def monitor_loop() -> None:
    index = 0
    warmup1 = True
    warmup2 = False
    safe_trace_file = config["SAFE_TRACE_FILE"]
    warmup_time = config["WARMUP_TIME"]
    anom_classifier = None
    formulae = []
    bin_classifier = None
    progress_bar(index=0, warmup_stage=1)
    anomaly_statuses = []
    while True:
        prompt = " " if warmup1 or warmup2 else ">"
        response = read_user_input(prompt)
        if response == "q":
            print("Exiting monitor...")
            exit()
        if response == "p":
            print_trees(bin_classifier=bin_classifier, anom_classifier=anom_classifier)
            continue
        if response == "w":
            show_weights(sensor_index=0)
            continue
        if response == "g":
            plot_graph()
            continue
        anomaly_info = read_anomaly_indices() if response == "a" and not (warmup1 or warmup2) else ("normal", np.array([]))
        anom_type, anomaly_indices = anomaly_info
        new_batch = get_new_batch(
            batch_size=config["BATCH_SIZE"],
            num_sensors=config["NUM_SENSORS"],
            index=index,
            source_file=config["SOURCE_FILE"],
        )
        index += 1
        if warmup1:
            progress_bar(warmup_stage=1, index=index)
            if index >= warmup_time:
                warmup1 = False
                warmup2 = True
                print("\nWarmup 1 complete.")
            write_new_batch(new_batch=new_batch, outfile=safe_trace_file)
            continue
        if warmup2:
            progress_bar(warmup_stage=2, index = index - warmup_time)
            if index >= warmup_time * 2:
                warmup2 = False
                print("\nWarmup complete.")
        test = preprocess("".join(new_batch), csv=False)
        train = preprocess(safe_trace_file)
        num_sensor_ids = train.shape[1] // 2
        indices_used = np.arange(num_sensor_ids) if config["PRESSURE"] else np.arange(num_sensor_ids, 2*num_sensor_ids)
        train = train[:, indices_used]
        test = test[:, indices_used]
        if anom_type == "small":
            test[:, anomaly_indices] += config["SMALL_ANOMALY_SIZE"]
        elif anom_type == "large":
            test[:, anomaly_indices] += config["LARGE_ANOMALY_SIZE"]
        elif anom_type == "all":
            test += config["LARGE_ANOMALY_SIZE"]
        num_evaluations = 2 # 27
        for sensor_index in range(num_evaluations):
            if not warmup2:
                print("Evaluating sensor index:", sensor_index + 1)
            elif len(anomaly_statuses) <= sensor_index:
                anomaly_statuses.append(False)
                formulae.append(None)
            residuals = get_residuals(
                train=train,
                test=test,
                sensor_index=sensor_index,
            )
            if not warmup2:
                print("Mean of residuals:", residuals.mean())
            new_trace = ",".join(residuals.astype(str))
            formula = formulae[sensor_index]
            if not new_batch_ok(residuals=residuals, formula=formula, new_batch=new_batch, sensor_index=sensor_index):
                confirmation, anom_classifier = log_anomaly(
                    new_batch, new_trace, sensor_index, anom_classifier, warmup2=warmup2
                )
                if confirmation:
                    start_anomaly(new_batch, sensor_index + 1)
                    anomaly_statuses[sensor_index] = True
                    formulae, bin_classifier = update_spec(
                        formulae=formulae,
                        sensor_index=sensor_index,
                        bin_classifier=bin_classifier,
                        new_trace=residuals,
                        new_label="Anomaly",
                    )
                    continue
            if sensor_index < len(anomaly_statuses) and anomaly_statuses[sensor_index]:
                end_anomaly(new_batch, sensor_index + 1)
                anomaly_statuses[sensor_index] = False
            write_new_batch(new_batch=new_batch, outfile=safe_trace_file)
            residuals_file = get_filename("residuals", sensor_index=sensor_index)
            with open(residuals_file, "a") as f:
                f.write("\n" + new_trace)
            if not warmup2:
                formulae, bin_classifier = update_spec(
                    formulae=formulae,
                    sensor_index=sensor_index,
                    bin_classifier=bin_classifier,
                    new_trace=residuals,
                    new_label="Safe",
                )



def main() -> None:
    write_header(config["SOURCE_FILE"], config["SAFE_TRACE_FILE"])
    clear_files()
    print_intro()
    monitor_loop()

if __name__ == "__main__":
    main()
