#!/bin/bash

# Define the path to your experiment folders
base_path="/teamspace/jobs"

# Define the destination path
destination_path="compare_datasets_exp"

# Debug: Print paths
echo "Base path: $base_path"
echo "Destination path: $destination_path"

# Ensure the destination path exists, create it if it doesn't
if [ ! -d "$destination_path" ]; then
    echo "Destination path $destination_path does not exist. Creating it."
    mkdir -p "$destination_path"
    
    # Check if the directory was created successfully
    if [ $? -ne 0 ]; then
        echo "Failed to create destination directory $destination_path"
        exit 1
    else
        echo "Successfully created $destination_path"
    fi
else
    echo "Destination path $destination_path already exists."
fi

# Find all directories in the base_path that match the pattern
#experiment_folders=$(find "$base_path" -maxdepth 1 -type d | grep -E '.*/cer-exp-[0-9]{2}-[a-zA-Z0-9]{5}$') #uniform CER
#experiment_folders=$(find "$base_path" -maxdepth 1 -type d | grep -E '.*/cer-[0-9]{1}-wer-[0-9]{2}-exp$') #CER-WER pairs
#experiment_folders=$(find "$base_path" -maxdepth 1 -type d | grep -i 'blend') #The blended synthetic data
#experiment_folders=$(find /teamspace/jobs -maxdepth 1 -type d -iname '*data2*') # the tokens per obs and data volume experimetn
experiment_folders=$(find "$base_path" -maxdepth 1 -type d | grep -i 'dataset') # comparing the data experiment

# Debug: List found folders
echo "Found experiment folders:"
echo "$experiment_folders"

# Lockfile to prevent simultaneous copies
lockfile=/tmp/copy.lock

# Loop over each experiment folder and copy the contents of cer_exp
for experiment in $experiment_folders; do
    source_path="${experiment}/work/$destination_path"
    
    echo "Checking source path: $source_path"
    
    if [ -d "$source_path" ]; then
        # Wait until lockfile is removed if it already exists
        while [ -e "$lockfile" ]; do
            sleep 1
        done

        # Create lockfile
        touch "$lockfile"

        echo "Copying from $source_path to $destination_path"
        rsync -av --progress "$source_path"/ "$destination_path"
        
        # Check if the copy was successful
        if [ $? -ne 0 ]; then
            echo "Failed to copy from $source_path to $destination_path"
            rm -f "$lockfile"  # Remove lockfile
            exit 1
        else
            echo "Successfully copied from $source_path to $destination_path"
        fi

        # Remove lockfile
        rm -f "$lockfile"
    else
        echo "Directory $source_path does not exist."
    fi
done

echo "Copying complete."

