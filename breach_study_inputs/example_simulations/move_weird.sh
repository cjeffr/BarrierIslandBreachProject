#!/bin/bash

# Specify the target directory
target_directory="../../defunct_runs/"

# Loop over all files in the current directory
for file in *; do
    # If the file name does not contain an underscore
    #if [[ $file != *_* ]]; then
    if [[ ($file == w* && $file != *_6) || $file == wd* ]]; then
    # Move the file to the target directory
        mv "$file" "$target_directory"
    fi
done

