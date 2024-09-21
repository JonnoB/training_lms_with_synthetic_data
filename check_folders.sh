#!/bin/bash

#This script checks to see what folders have the work subfolder or not. This is because there are some missing files and I don't know why

# Directory to search
search_dir="$1"

# Initialize counters
contains_work=0
does_not_contain_work=0

# Loop over each subfolder in the directory
for dir in "$search_dir"/*/; do
    if [ -d "$dir/work/data_length_exp" ]; then
        contains_work=$((contains_work + 1))
    else
        does_not_contain_work=$((does_not_contain_work + 1))
    fi
done

# Output the results
echo "Folders containing 'work': $contains_work"
echo "Folders without 'work': $does_not_contain_work"


