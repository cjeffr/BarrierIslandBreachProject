#!/bin/bash
#!/bin/bash

# Specify the output file
output_file="done.txt"

# Loop through all directories in the current directory
for dir in */; do
    # If a subdirectory named _output does not exist
    if [  -d "${dir}_output" ]; then
        # Append the directory name to the output file
        mv "$dir" finished/
    fi
done
