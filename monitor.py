print("Loading resources...")

from ui import read_user_input, read_anomaly_indices, print_trees, progress_bar, show_weights, plot_graph, print_intro
from file_io import clear_files, write_header, get_new_batch, write_new_batch, end_anomaly, start_anomaly, get_filename
from model import get_residuals, update_spec, log_anomaly, new_batch_ok, apply_anomaly
from preproc import preprocess
import numpy as np
import json
import copy



with open("config.json", 'r') as config_file:
    config = json.load(config_file)

def monitor_loop() -> None:
    index = 0
    warmup1 = True
    warmup2 = False
    safe_trace_file = config["SAFE_TRACE_FILE"]
    warmup_time = config["WARMUP_TIME"]
    pressure_anom_classifier = None
    temp_anom_classifier = None
    anom_classifier = None
    formulae = []
    temp_formulae = []
    pressure_formulae = []
    pressure_bin_classifier = None
    temp_bin_classifier = None
    bin_classifier = None
    progress_bar(index=0, warmup_stage=1)
    pressure_anomaly_statuses = []
    temp_anomaly_statuses = []
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
        for sensor_type in ["PRESSURE", "TEMPERATURE"]:
            if sensor_type == "PRESSURE":
                temp_anom_classifier = copy.deepcopy(anom_classifier)
                temp_bin_classifier = copy.deepcopy(bin_classifier)
                temp_formulae = copy.deepcopy(formulae)
                temp_anomaly_statuses = anomaly_statuses.copy()
                anomaly_statuses = pressure_anomaly_statuses
                anom_classifier = pressure_anom_classifier
                bin_classifier = pressure_bin_classifier
                formulae = pressure_formulae
            else:
                pressure_anom_classifier = copy.deepcopy(anom_classifier)
                pressure_bin_classifier = copy.deepcopy(bin_classifier)
                pressure_formulae = copy.deepcopy(formulae)
                pressure_anomaly_statuses = anomaly_statuses.copy()
                anom_classifier = temp_anom_classifier
                bin_classifier = temp_bin_classifier
                anomaly_statuses = temp_anomaly_statuses
                formulae = temp_formulae
            ## Code using anom_classifier and bin_classifier
            indices_used = np.arange(num_sensor_ids) if sensor_type=="PRESSURE" else np.arange(num_sensor_ids, 2*num_sensor_ids)
            train_used = train[:, indices_used]
            test = apply_anomaly(data=test, anomaly_indices=anomaly_indices, anom_type=anom_type)
            test_used = test[:, indices_used]
            num_evaluations = 2 # 27
            for sensor_index in range(num_evaluations):
                if not warmup2:
                    print(f"{sensor_type}: Evaluating sensor index {sensor_index + 1}")
                elif len(anomaly_statuses) <= sensor_index:
                    anomaly_statuses.append(False)
                    formulae.append(None)
                residuals = get_residuals(
                    train=train_used,
                    test=test_used,
                    sensor_index=sensor_index,
                    sensor_type=sensor_type,
                )
                if not warmup2:
                    print("Mean of residuals:", residuals.mean())
                new_trace = ",".join(residuals.astype(str))
                formula = formulae[sensor_index]
                if not new_batch_ok(residuals=residuals, formula=formula, new_batch=new_batch, sensor_index=sensor_index, sensor_type=sensor_type):
                    confirmation, anom_classifier = log_anomaly(
                        new_batch, new_trace, sensor_index, anom_classifier, warmup2=warmup2, sensor_type=sensor_type
                    )
                    if confirmation:
                        if not anomaly_statuses[sensor_index]:
                            start_anomaly(new_batch, sensor_index + 1)
                        anomaly_statuses[sensor_index] = True
                        formulae, bin_classifier = update_spec(
                            formulae=formulae,
                            sensor_index=sensor_index,
                            bin_classifier=bin_classifier,
                            new_trace=residuals,
                            new_label="Anomaly",
                            sensor_type=sensor_type,
                        )
                        continue
                if sensor_index < len(anomaly_statuses) and anomaly_statuses[sensor_index]:
                    end_anomaly(new_batch, sensor_index + 1)
                    anomaly_statuses[sensor_index] = False
                write_new_batch(new_batch=new_batch, outfile=safe_trace_file)
                residuals_file = get_filename("residuals", sensor_index=sensor_index, sensor_type=sensor_type)
                with open(residuals_file, "a") as f:
                    f.write("\n" + new_trace)
                if not warmup2:
                    formulae, bin_classifier = update_spec(
                        formulae=formulae,
                        sensor_index=sensor_index,
                        bin_classifier=bin_classifier,
                        new_trace=residuals,
                        new_label="Safe",
                        sensor_type=sensor_type,
                    )



def main() -> None:
    write_header(config["SOURCE_FILE"], config["SAFE_TRACE_FILE"])
    clear_files()
    print_intro()
    monitor_loop()

if __name__ == "__main__":
    main()
