#!/bin/bash

# Set the path to the main directory where you want to search for subdirectories
MAIN_DIR="./"

# Traverse through all subdirectories
for d in "$MAIN_DIR"/*; do
    if [ -d "$d" ]; then
        # Check if the subdirectory contains the specified file
        if ! ls "$d"/_output/gauge*.txt &>/dev/null; then
            # Move the subdirectory to 'no_gauge_output'
            mv "$d" "$MAIN_DIR/no_gauge_output/"
        fi
    fi
done

echo "Folders without gauge output files have been moved to 'no_gauge_output'."

