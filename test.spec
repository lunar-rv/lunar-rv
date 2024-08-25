# The file containing the raw sensor values
input = "inputs/test.csv"

# Necessary constants for building a monitor
safe = 10
batch = 4

# STL formula templates
add type pressure 2
add type temperature 3
add stl G_avg
add stl F
add stl G