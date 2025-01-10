#!/bin/bash
set -e  # Exit immediately if any command exits with a non-zero status

source /hps/software/users/literature/textmining-ml/envs/ml_tm_pipeline_env/bin/activate

ENV_FILE="/hps/software/users/literature/textmining-ml/.env_paths"

if [ -f "$ENV_FILE" ]; then
    # Try to source the environment file
    if source "$ENV_FILE"; then
        echo "SECTION_TAGGER_WORKER:Loaded environment file from $ENV_FILE"
    else
        echo "Error: SECTION_TAGGER_WORKER:Could not load environment file from $ENV_FILE"
        exit 1
    fi
else
    echo "Error: SECTION_TAGGER_WORKER:Environment file not found at $ENV_FILE"
    exit 1
fi


# Parameters from the main script
FULLTEXT_SOURCE_DIR="$1"
TODAY_OUTPUT_DIR="$2"
YESTERDAY_OUTPUT="$3"

# Define the Python section tagger script path using LIB_PATH from .env_paths
SECTION_TAGGER_SCRIPT="${LIB_PATH}/python_scripts/section_sentence_tagger.py"

# Retrieve the correct file based on the SLURM_ARRAY_TASK_ID
file=$(find "$FULLTEXT_SOURCE_DIR" -type f -name "patch-*.xml.gz" | sort -V | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")

# Extract file number for naming output
file_number=$(echo "$file" | grep -oP '(?<=patch-\d{2}-\d{2}-\d{4}-)\d+(?=\.xml\.gz)')

# Define the output file path
output_file="${TODAY_OUTPUT_DIR}/fulltext/sections/patch-${YESTERDAY_OUTPUT}-${file_number}.jsonl.gz"

# Debugging: Print paths to confirm correctness
echo "Running Python script with paths:"
echo "SECTION_TAGGER_SCRIPT: $SECTION_TAGGER_SCRIPT"
echo "Input file: $file"
echo "Output file: $output_file"

# Run the section tagger script on the file
if ! python "$SECTION_TAGGER_SCRIPT" --input "$file" --output "$output_file" --type f; then
    echo "Error: Failed to run section tagger on file $file" >&2
    exit 1  # Exit with error status if Python script fails
fi

echo "Processed file $file and saved output to $output_file"

