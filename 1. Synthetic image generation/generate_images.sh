#! /usr/bin/bash

# Start timer
START_TIME=$(date +%s)

# Define paths
SOURCE_FOLDER="all_signals"
TEMP_FOLDER="temp_signals"
OUTPUT_FOLDER="ecg_signals"
PROCESSED_FOLDER="processed_signals"

# Create necessary directories if they don't exist
mkdir -p "$TEMP_FOLDER"
mkdir -p "$OUTPUT_FOLDER"
mkdir -p "$PROCESSED_FOLDER"

# Number of images to generates
NUM_IM=10
NUM_FILES=$((NUM_IM * 2))

# ---------------------

# Step 1: Move files from source to temp folder
echo "Moving $NUM_FILES files from $SOURCE_FOLDER to $TEMP_FOLDER..."
ls "$SOURCE_FOLDER" | head -"$NUM_FILES" | while read file; do
  mv "$SOURCE_FOLDER/$file" "$TEMP_FOLDER"
done

echo "$NUM_FILES files moved to $TEMP_FOLDER."

# Step 2: Run the Python command to process the files
echo "Running the Python processing command..."

nice python gen_ecg_images_from_data_batch.py \
  -i "$TEMP_FOLDER" \
  -o "$OUTPUT_FOLDER" \
  -se 1 \
  --mask_unplotted_samples \
  --max_num_images "$NUM_IM" \
  --random_resolution \
  --random_grid_present 0.7 \
  --random_print_header 1 \
  --random_bw 1 \
  --random_grid_color \
  --lead_name_bbox \
  --lead_bbox \
  --store_config \
  --num_columns 3 \
  --augment \
  --wrinkles

echo "Processing completed."

# Step 3: Move the original files from temp to the processed folder
echo "Moving processed files from $TEMP_FOLDER to $PROCESSED_FOLDER..."
mv "$TEMP_FOLDER"/* "$PROCESSED_FOLDER"

# Cleanup temp folder
echo "Cleaning up $TEMP_FOLDER..."
rm -rf "$TEMP_FOLDER"/*

# ---------------------------------

# End timer
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

# Output the time taken
echo "Script completed successfully in $((ELAPSED_TIME / 60)) minutes and $((ELAPSED_TIME % 60)) seconds."