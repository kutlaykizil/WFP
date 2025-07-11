#!/bin/bash

# Function to extract the .tif file
extract_tif() {
  local tar_path="$1"
  local output_dir="$2"
  local temp_dir="temp_extract"

  # Extract the tar archive
  tar -xf "$tar_path" -C "$temp_dir"

  # Find the .tif file (adjust the glob pattern if needed)
  tif_path=$(find $temp_dir/Coper*/DEM/*.tif)

  # Copy the .tif file to the output directory
  cp "$tif_path" "$output_dir"

  # Clean up the temporary directory
  rm -r $temp_dir/*
}

# Set the input and output directories
tar_dir="/home/kutlay/Turkey/"
output_dir="/home/kutlay/DEM"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Process each .tar file in the input directory
for filename in "$tar_dir"/*.tar; do
  echo "Processing $filename..."
  extract_tif "$filename" "$output_dir"
  echo "Finished processing $filename"
done

echo "Extraction complete!"