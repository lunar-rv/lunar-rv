## License

See *LICENSE.md* for details

## Installation

To set up the tool, carry out the following:

```bash
# Clone the repository
git clone git@github.com:julius-gasson/lunar.git

# Navigate to the project directory
cd lunar

# Install dependencies
pip install -r requirements.txt
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
- *TREE_CONFIG -> MAX_DEPTH*: the maximum depth of the anoWmaly classification trees
- *TREE_CONFIG -> BETA*: a constant which determines the relative importance of STL-specific and traditional entropy measures. See report for details.
- *WARMUP_ANOMALIES*: number of anomalies that need to be processed before the monitor switches to a binary classification approach
- *ONLINE*: If this is set to *true*, the monitor is run in online mode - new batches arrive upon being prompted by the user; the user can 
- *WINDOW_SIZE*: the rolling window size used for linear regression.

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

For an example of an acceptable specification, see *test.spec* and *gas_network.spec*.
## Input formatting
- Inputs should be presented in CSV (comma-separated variables) format without column headers.
- Each column contains all the readings for a particular sensor, ordered from earliest to most recent.
- All the readings for a particular sensor type are concatenated into a block of shape *(num_readings, num_sensors)*
- All the blocks are concatenated horizontally into a full dataset of shape *(num_readings, total_num_sensors)*
- Finally, columns N+1 and N+2 contain the date and the time of the set of N sensor readings respectively, in the format *dd/mm/yyyy*,*hh/mm/ss*
- There must not be any missing sensor values.
- For an example of an acceptable input format with 27 pressure sensors and 27 temperature sensors, see *inputs/traces.csv*

## Anomaly types
- Notes:
    - Each sensor in the network has its own anomaly detection formula and classification tree
The STL formula information for anomaly detection is to be interpreted as follows:
-  G/always:
![G graph][./examples/G.png]
    - Here, the overall formula includes a *G* subformula which requires that the residuals should be below a certain value (here, 0.554695...) at all times. This was violated twice between 01:00 and 02:00. The shaded area indicates the batch of 96 readings (= 24 hours) on which this formula was evaluated. 
    - This operator is useful for anomalies characterised by a short, large spike, as only one value above the *G* threshold is required for an anomaly to be flagged.

- F/eventually:
![F graph][./examples/F.png]
    - The F_[0, *b*) subformula describes the property that residuals should be below a certain value at one point (at least) in every period of *b* time steps. If a time step in the dataset is 15 minutes, F_[0, 77) error < 0.208015... means that the residual size should be less than 0.208015... at least once in every period of length 19h 15m.
    - The key period for which the relational formula (i.e. error < 0.208015...) is violated actually lasts for 140 time steps, or 35 hours. On the graph, the whole 35-hour period is shaded, even though a violation would also be detected if this period was shorter. 
    - Other violation periods which did not last long enough to cause a violation of the *F* subformula are not shaded as they do not explain why an anomaly was detected.
    - This operator is important for detecting anomalies which last for a prolonged period of time.

- G_avg/G_bar/mean:
![G_avg graph][./examples/G_avg.png]
    - G_[0, 80) error < 1.51603... means that the size of the error must *on average* be below 1.51603... during every period of 80 consecutive time steps (here, 20 hours). 
    - The shading of the period from 00:45 to 23:45 indicates that the *G_avg* subformula was violated during the periods 00:45-20:45, 01:00-21:00, 01:15-21:15... 3:45-23:45. 
    - This operator also helps to detect longer-lasting anomalies, and is more robust to noise than the *F* operator.
    - In some cases, there might be multiple periods of violation within the batch

![Graph showing overlap][./examples/overlap.png]


