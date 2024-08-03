import json
import numpy as np

with open("config.json", "r") as config_file:
    config = json.load(config_file)

def clear_files(*args) -> None:
    for filename in args:
        with open(filename, "w"):
            pass

def write_header(source_file, safe_trace_file) -> None:
    with open(source_file, "r") as s:
        header = s.readlines()[0]
        with open(safe_trace_file, "w") as i:
            i.write(header)

def get_new_batch(
    batch_size=config["BATCH_SIZE"],
    num_sensors=54,
    index=0,
    source_file=config["SOURCE_FILE"],
) -> list:
    start = index * batch_size * num_sensors + 1
    end = (index + 1) * batch_size * num_sensors + 1
    with open(source_file, "r") as s:
        lines = s.readlines()
        if end >= len(lines):
            return False
        values_to_add = lines[start:end]
        return values_to_add
    

def write_new_batch(new_batch, outfile) -> None:
    with open(outfile, "a") as i:
        i.write("\n")
        i.writelines(new_batch)

def write_weights(model, filename=config["WEIGHTS_FILE"]) -> None:
    weights = model.coef_[:26]
    np.savetxt(filename, weights, delimiter=",")

def end_anomaly(new_batch: list, sensor_index: int):
    first_reading = new_batch[0]
    date = first_reading.split(";")[1]
    with open(config["LOG_FILE"], "a") as log:
        log.write(f"Anomaly at sensor {sensor_index} resolved at {date}\n")
    exit()

def start_anomaly(new_batch: list, sensor_index: int):
    first_reading = new_batch[0]
    date = first_reading.split(";")[1]
    with open(config["LOG_FILE"], "a") as log:
        log.write(f"Anomaly at sensor {sensor_index} detected at {date}\n")