## License

See *LICENSE.md* for details

## Installation

To set up the tool, carry out the following:

```bash
# Clone the repository
git clone https://github.com/jg2023/monitor.git

# Navigate to the project directory
cd monitor

# Install dependencies
pip install -r requirements.txt
```

It may also be necessary to add this directory to PYTHONPATH:
```bash
# Export directories to path
export PYTHONPATH=$PYTHONPATH:path/to/monitor
```

This repository contains:
- *gas_network.spec*: example specification for provided gas network data
- *test.spec*: toy example specification.
- *src*: directory containing monitor source code.
- *inputs*: directory containing example input traces.
- *outputs*: directory in which monitor information is saved. This includes all prediction model weights, anomalies detected, anomaly graphs, residuals of safe traces and anomaly detection formulae.
- *config.json*: configuration options for the monitor.
## Instructions for use
- To build and run a monitor, use the command:
```bash
python src/rv.py {specification_file}
```

## Monitor configuration

Various options for fine-tuning the monitor configuration are found in *config.json*.
- *WARMUP_1_PROPORTION*: the monitor has two warmup stages: in stage 1, new readings are only used to train the prediction models. In stage 2, the monitor learns the typical patterns of residuals in the dataset as well. This constant determines the proportion of the 'safe' period to be spent during stage 1
- *PLOT_ANOMALY_GRAPHS*: when set to true, the monitor will display a graph of the sensor values and the residuals over a batch, whenever an anomaly is detected.
- Directories in which to save monitor outputs
- *GAMMA*: controls the extent to which tighter F intervals are rewarded when evaluating formulae for anomaly detection
- *EPSILON_COEF*: this constant, multiplied by the standard deviation of the cluster of robustness values for safe traces, determines the distance of the decision boundary from the least robust data point
- *BIDIRECTIONAL_ONLY*: when this option is set to true, the sensor connection graphs will only display an edge *IJ* if Sensor I contributes significantly to Sensor J and vice versa
- *TREE_CONFIG -> MAX_DEPTH*: the maximum depth of the anomaly classification trees
- *TREE_CONFIG -> BETA*: a constant which determines the relative importance of STL-specific and traditional entropy measures. See report for details.
- *WARMUP_ANOMALIES*: number of anomalies that need to be processed before the monitor switches to a binary classification approach
- *ONLINE*: If this is set to *true*, the monitor is run in online mode - new batches arrive upon being prompted by the user; the user can 
- *PLOT_RESIDUALS_GRAPHS*: in offline mode, this provides an option to display the predicted values against the actual values of the data over time.
Offline mode is for evaluating the program.
- *ADD_ANOMALIES_OFFLINE*: add synthetic anomalies to the data when running the monitor offline. This makes it possible to evaluate the model's ability to recall actual anomalies.

## Specification language guidelines
- *input = \<path/to/input/file\>*: the file containing the input dataset (formatting guidelines are below)
- *Necessary constants*
    - *safe = \<integer\>*: the number of initial traces that are classified as safe, which are used by the monitor to learn sensor patterns
    - *batch = \<integer\>*: the number of time intervals to process per batch
- *add stl \<F/G/G_avg\>*
    - Adds a temporal logic operator (*F*, *G* or *G_avg*) to the anomaly detection template.
    - At least one operator must be selected.
- *add type \<reading_type\> \<num_sensors\>*
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
- Finally, columns N+1 and N+2 contain the date and the time of the set of N sensor readings respectively, in the format *dd/mm/yyyy*,*hh/mm/ss*
- There must not be any missing sensor values.
- For an example of an acceptable input format with 27 pressure sensors and 27 temperature sensors, see *inputs/traces.csv*
