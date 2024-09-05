#!/bin/bash

# Generate hostnames from texel01 to texel44
hosts=()
for i in $(seq -f "texel%02g" 1 23)
do
  hosts+=("$i")
done

for i in $(seq -f "texel%02g" 25 44)
do
  hosts+=("$i")
done

# Generate hostnames from oak01 to oak38
for i in $(seq -f "oak%02g" 7 20)
do
  hosts+=("$i")
done

# Initialize the index
index=0
host_index=0
# Loop through each host and run the SSH command
echo "command used: nohup bash -c 'source monitor.sh && python src/rv.py gas_network.spec 4 $index $1' >> output.log 2>&1 &"
while [ $index -lt 54 ]; do
    echo "index: $index"
    
    # Access the current host from the array
    host=${hosts[$host_index]}
    # ssh -o StrictHostKeyChecking=no "$host" "pkill -f python"
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=7  "$host" "nohup bash -c 'source monitor.sh && python src/rv.py gas_network.spec 4 --index $index --model $1' >> output.log 2>&1 &"
    if [ $? -ne 0 ]; then
        echo "Error: Connection to $host failed (e.g., Connection timed out)"
    else
      index=$((index + 1))
    fi
    host_index=$((host_index + 1))
done


