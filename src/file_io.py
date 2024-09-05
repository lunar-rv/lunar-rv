print("Loading file i/o functions...")
import json
import numpy as np
import os

with open("config.json", "r") as config_file:
    config = json.load(config_file)

def get_filename(output_type: str, sensor_index: int, sensor_type: str, suffix=".csv", remove_plural=False) -> str:
    output_dir = config[output_type.upper() + "_DIR"]
    output_type = output_type[:-1] if remove_plural else output_type
    return f"{output_dir}/{sensor_type.lower()}/sensor_{sensor_index+1}_{output_type}{suffix}"
    

def clear_files(reading_types) -> None:
    output_directories = [
        config["WEIGHTS_DIR"], 
        config["RESIDUALS_DIR"], 
        config["ANOMALIES_DIR"],
        config["SPECS_DIR"],
        config["GRAPH_DIR"],
    ]
    for dir in output_directories:
        if not os.path.exists(dir):
            os.mkdir(dir)
        for t in reading_types:
            subdir = os.path.join(dir, t)
            if os.path.exists(subdir):
                for file in os.listdir(subdir):
                    full_filename = os.path.join(subdir, file)
                    with open(full_filename, "w"):
                        pass
            else:
                os.mkdir(subdir)
    with open(config["SAFE_TRACE_FILE"], "w"):
        pass
    with open(config["LOG_FILE"], "w"):
        pass
    
def write_header(source_file, safe_trace_file) -> None:
    with open(source_file, "r") as s:
        header = s.readlines()[0]
        with open(safe_trace_file, "w") as i:
            i.write(header)

### THIS FUNCTION WOULD NOT BE NEEDED IN PRACTICE
def get_new_batch(
    batch_size,
    source_file,
    index=0,
) -> list:
    start = index * batch_size
    end = (index + 1) * batch_size
    with open(source_file, "r") as s:
        lines = s.readlines()
        if end >= len(lines):
            print("End of file has been reached!")
            print("Exiting monitor...")
            exit()
        values_to_add = lines[start:end]
        return values_to_add
    

def write_new_batch(new_batch, outfile) -> None:
    with open(outfile, "a") as i:
        i.writelines(new_batch)

def write_weights(model, sensor_type) -> None:
    sensor_index = model.sensor_index
    weights = model.coef_
    indices = model.sensors_used
    filename = get_filename("weights", sensor_index, sensor_type=sensor_type)
    with open(filename, 'w') as f:
        f.write(",".join(map(str, indices)) + "\n")
        np.savetxt(f, weights[None], delimiter=",", fmt='%.6f')

def end_anomaly(new_batch: list, sensor_index: int) -> None:
    first_reading = new_batch[0]
    date = first_reading.split(",")[-2]
    with open(config["LOG_FILE"], "a") as log:
        log.write(f"Anomaly at sensor {sensor_index} resolved at {date}\n")

def start_anomaly(new_batch: list, sensor_index: int) -> None:
    first_reading = new_batch[0]
    date = first_reading.split(",")[-2]
    with open(config["LOG_FILE"], "a") as log:
        log.write(f"Anomaly at sensor {sensor_index} detected at {date}\n")