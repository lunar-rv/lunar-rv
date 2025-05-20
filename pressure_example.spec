# The file containing the raw sensor values
# input = "inputs/pressure_traces.csv"
input = "inputs/with_anomalies.csv"

# Necessary constants for building a monitor
safe = 20
batch = 96

# STL formula templates
add type pressure 25
add stl F
add stl G_avg
add stl G
