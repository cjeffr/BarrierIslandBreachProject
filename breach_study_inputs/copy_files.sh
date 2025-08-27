#!/bin/bash

# Specify the files you want to copy
files_to_copy=("setrun.py" "setplot.py" "mapper.py" "Makefile" "setprob.f90" "breach_module.f90" "b4step2.f90")

# Iterate through all subfolders and copy the files
for folder in */; do
    folder="${folder%/}"  # Remove trailing slash
    echo "Copying files into $folder..."
    
    
    # Copy the files into the subfolder
    cp "${files_to_copy[@]}" "$folder"
done

echo "Copying files into all subfolders completed."

