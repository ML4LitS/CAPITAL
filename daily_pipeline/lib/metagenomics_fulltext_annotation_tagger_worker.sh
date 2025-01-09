#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=8G
#SBATCH --time=10:00:00

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Load your environment
source /hps/software/users/literature/textmining-ml/envs/ml_tm_pipeline_env/bin/activate


ENV_FILE="/hps/software/users/literature/textmining-ml/.env_paths"

if [ -f "$ENV_FILE" ]; then
    # Try to source the environment file
    if source "$ENV_FILE"; then
        echo "FULLTEXT_ANNOTATION_TAGGER_WORKER:Loaded environment file from $ENV_FILE"
    else
        echo "Error: FULLTEXT_ANNOTATION_TAGGER_WORKER:Could not load environment file from $ENV_FILE"
        exit 1
    fi
else
    echo "Error: FULLTEXT_ANNOTATION_TAGGER_WORKER:Environment file not found at $ENV_FILE"
    exit 1
fi


# Get input arguments
TODAY_OUTPUT_DIR="$1"
#MODEL_PATH_QUANTIZED="$2"

# Get the list of section-tagged files
SECTION_FILES=($(find "${TODAY_OUTPUT_DIR}/fulltext/sections" -type f -name "*.jsonl.gz" | sort))
INPUT_FILE="${SECTION_FILES[$SLURM_ARRAY_TASK_ID]}"

echo "Annotating full-text file: $INPUT_FILE"

# Define output directory for annotations
OUTPUT_DIR="${TODAY_OUTPUT_DIR}/fulltext/annotations/metagenomics"

# Run your full-text annotation script
python "${LIB_PATH}/python_scripts/annotation_tagger_linker_metagenomics.py" \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_DIR" 
   # --model_path "$MODEL_PATH_QUANTIZED"

echo "Metagenomics Full-text annotation tagging completed for file: $INPUT_FILE"

