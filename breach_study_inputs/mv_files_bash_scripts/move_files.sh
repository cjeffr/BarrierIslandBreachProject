#!/bin/bash

# Assuming file_list.txt contains the names of the files you want to copy
while IFS= read -r filename; do
    # Use find to get a list of subdirectories (excluding files) in the current directory
    for folder in $(find . -maxdepth 1 -type d ! -name '.' | sed 's|^\./||'); do
        # Check if the file exists in the current folder
        if [ -e "$folder/$filename" ]; then
            # Create the destination directory if it doesn't exist
            mkdir -p "../../barrier_breach_optimization_trials/amr_sensitivity/$folder"
            
            # Copy the file to the destination folder maintaining the directory structure
            cp "$folder/$filename" "../../barrier_breach_optimization_trials/amr_sensitivity/$folder/"
            
            echo "Copied $filename from $folder to Destination/$folder/"
        fi
    done
done < file_list.txt

