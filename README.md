# Project Name

## Installation

Step-by-step instructions on how to install the product.

```bash
# Clone the repository
git clone https://github.com/jg2023/monitor.git

# Navigate to the project directory
cd monitor

# Install dependencies
pip install -r requirements.txt
```

This repository contains:
- *spec.file*: example specification
- *src*: directory containing monitor source code
- *inputs*: directory containing example 
- *outputs*: directory in which monitor logs are saved

## Instructions for use
- To build and run a monitor, use the command:
```bash
python rv.py {specification_file}
```

## Specification language guidelines
- *input = <path/to/input/file>*: the file containing the input dataset (formatting guidelines are below)
- *Necessary constants*
    - *safe = <integer>*: the number of initial traces that are classified as safe, which are used by the monitor to learn sensor patterns
    - *batch = <integer>*: the number of time intervals to process per batch
- *add stl <F/G/G_avg>*
    - Adds a temporal logic operator (*F*, *G* or *G_avg*) to the anomaly detection template.
    - At least one operator must be selected.
- *add type <reading_type> <num_sensors>*
    - Adds a new reading type to evaluate from the traces.
    - At least one type must be selected.
    - This is followed by an integer argument which represents the number of sensors for that reading type in the input dataset. For example, *add type pressure 27* will classify 27 columns in the dataset as pressure readings.
    - Reading types in the dataset are specified from left to right.

For an example of an acceptable specification, see *spec.file*.
## Input formatting
- Inputs should be presented in CSV (comma-separated variables) format without column headers.
- Each column contains all the readings for a particular sensor, ordered from earliest to most recent.
- All the readings for a particular sensor type are concatenated into a block of shape *(num_readings, num_sensors)*
- All the blocks are concatenated horizontally into a full dataset of shape *(num_readings, total_num_sensors)*
 Columns N+1 and N+2 contain the date and the time of the set of N sensor readings respectively, in the format *dd/mm/yyyy*,*hh/mm/ss*
- There must not be any missing sensor values.
- For an example of an acceptable input format with 27 pressure sensors and 27 temperature sensors, see *inputs/traces.csv*
