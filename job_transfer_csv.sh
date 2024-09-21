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
#experiment_folders=$(find "$base_path" -maxdepth 1 -type d -iname '*data2*')
experiment_folders=$(find "$base_path" -maxdepth 1 -type d | grep -i 'datasets')
# Debug: List found folders
echo "Found experiment folders:"
echo "$experiment_folders"

# Lockfile to prevent simultaneous copies
lockfile=/tmp/copy.lock

# Loop over each experiment folder and copy the CSV file
for experiment in $experiment_folders; do
    source_path="${experiment}/work/$destination_path/results"
    
    echo "Checking source path: $source_path"
    
    if [ -d "$source_path" ]; then
        # Wait until lockfile is removed if it already exists
        while [ -e "$lockfile" ]; do
            sleep 1
        done

        # Create lockfile
        touch "$lockfile"

        # Extract the experiment folder name (e.g., data2-obs-1024-token-length-100-exp)
        experiment_name=$(basename "$experiment")
        
        # Find the CSV file in the results folder (assuming only one CSV exists)
        csv_file=$(find "$source_path" -maxdepth 1 -type f -name '*.csv')
        
        if [ -n "$csv_file" ]; then
            # Define the new file name based on the experiment name
            new_csv_name="${experiment_name}.csv"
            
            # Copy the CSV file to the destination folder using rsync
            echo "Copying and renaming $csv_file to $destination_path/$new_csv_name"
            rsync -av "$csv_file" "$destination_path/$new_csv_name"
            
            # Check if the copy was successful
            if [ $? -ne 0 ]; then
                echo "Failed to copy $csv_file to $destination_path/$new_csv_name"
                rm -f "$lockfile"  # Remove lockfile
                exit 1
            else
                echo "Successfully copied and renamed $csv_file to $destination_path/$new_csv_name"
            fi
        else
            echo "No CSV file found in $source_path."
        fi

        # Remove lockfile
        rm -f "$lockfile"
    else
        echo "Directory $source_path does not exist."
    fi
done

echo "CSV files copied and renamed."
