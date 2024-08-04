import json
import numpy as np
import os

def get_filename(output_type: str, sensor_index: int, suffix=".csv", remove_plural=False) -> str:
    output_dir = config[output_type.upper() + "_DIR"]
    output_type = output_type[:-1] if remove_plural else output_type
    return output_dir + f"/sensor_{sensor_index+1}_{output_type}{suffix}"
    
with open("config.json", "r") as config_file:
    config = json.load(config_file)

def clear_files() -> None:
    output_types = ["residuals", "anomalies", "specs"]
    for ot in output_types:
        # for filename in os.listdir(output_dir):
        #     with open(filename, "w"):
        #         pass
        if ot == "specs":
            filenames = [get_filename(ot, i, suffix=".stl", remove_plural=True) for i in range(config["NUM_SENSORS"])]
        else:
            filenames = [get_filename(ot, i) for i in range(config["NUM_SENSORS"] // 2)]
        for filename in filenames:
            with open(filename, "w"):
                pass
    with open(config["LOG_FILE"], "w"):
        pass

def write_header(source_file, safe_trace_file) -> None:
    with open(source_file, "r") as s:
        header = s.readlines()[0]
        with open(safe_trace_file, "w") as i:
            i.write(header)

def get_new_batch(
    batch_size=config["BATCH_SIZE"],
    num_sensors=config["NUM_SENSORS"],
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

def write_weights(model) -> None:
    sensor_index = model.sensor_index
    weights = model.coef_
    indices = model.sensors_used
    filename = get_filename("weights", sensor_index)
    with open(filename, 'w') as f:
        f.write(",".join(map(str, indices)) + "\n")#
        np.savetxt(f, weights[None], delimiter=",", fmt='%.6f')

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