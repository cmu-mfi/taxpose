#!/bin/bash

# This script will run 3 commands in succession. We'll specify each command as a separate argument to the script.
# Importantly, we'll want to augment the output of commands 2 and 3 with some output generated by command 1 (which returns some stdout value).

# Get the first command (from the first argument)
cmd1=$1
# Get the second command (from the second argument)
cmd2=$2
# Get the third command (from the third argument)
cmd3=$3


# Run the first command and store the output in a variable
# The output of the first command will be used as an argument to the second and third commands


tmpfile=$(mktemp)

# Execute the command, tee the output to both stdout and the temporary file.
$cmd1 | tee "$tmpfile"

cmd_status=$?

# Check if the command failed.
echo "Command status: $cmd_status"
if [ $cmd_status -ne 0 ]; then
    echo "Command failed with status $cmd_status"
    exit $cmd_status
fi

# Read the temporary file into a variable.
output1=$(<"$tmpfile")

# output1=$($cmd1 | tee >(cat))

# parse the output of the first command, we're looking for something which looks like: "Run ID: d5ymul7w", and we want to extract the "d5ymul7w" part
run_id=$(echo $output1 | grep -oP 'Run ID: \K\w+')

# Run the second command, passing output1 as an argument
echo $cmd2 checkpoints.ckpt_file=r-pad/taxpose/model-${run_id}:v0
$cmd2 checkpoints.ckpt_file=r-pad/taxpose/model-${run_id}:v0

# Run the third command, passing output1 as an argument
echo $cmd3 +checkpoints.single_model_override=r-pad/taxpose/model-${run_id}:v0
$cmd3 +checkpoints.single_model_override=r-pad/taxpose/model-${run_id}:v0
