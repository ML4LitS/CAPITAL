#!/bin/bash

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

################################
# ./your_script.sh "2024-10-28"
#################################

source /hps/software/users/literature/textmining-ml/envs/ml_tm_pipeline_env/bin/activate

ENV_FILE="/hps/software/users/literature/textmining-ml/.env_paths"

if [ -f "$ENV_FILE" ]; then
    # Try to source the environment file
    if source "$ENV_FILE"; then
        echo "MAIN_PIPELINE:Loaded environment file from $ENV_FILE"
    else
        echo "Error: MAIN_PIPELINE:Could not load environment file from $ENV_FILE"
        exit 1
    fi
else
    echo "Error: MAIN_PIPELINE:Environment file not found at $ENV_FILE"
    exit 1
fi


# Define paths, dates, and timestamps 
CUSTOM_DATE="$1"  # The first command-line argument is treated as today's date
TODAY=$(date -d "$CUSTOM_DATE" +"%d_%m_%Y")
TODAY_OUTPUT=$(date -d "$CUSTOM_DATE" +"%Y_%m_%d")
YESTERDAY_OUTPUT=$(date -d "$CUSTOM_DATE -1 day" +"%Y_%m_%d")

# Define base directories
ABSTRACT_SOURCE_DIR="${BASE_DIR}/${TODAY}/abstract/source"
FULLTEXT_SOURCE_DIR="${BASE_DIR}/${TODAY}/fulltext/source"
TODAY_OUTPUT_DIR="${OUTPUT_BASE_DIR}/${TODAY_OUTPUT}"

# Define logs
FULLTEXT_LOG_DIR="${TODAY_OUTPUT_DIR}/fulltext/logs"
ABSTRACT_LOG_DIR="${TODAY_OUTPUT_DIR}/abstract/logs"

#Define script paths

SECTION_TAGGER_PATH="${LIB_PATH}/fulltext_section_tagger_worker.sh"
FULLTEXT_TAGGER_PATH="${LIB_PATH}/fulltext_annotation_tagger_worker.sh"
ABSTRACT_TAGGER_PATH="${LIB_PATH}/abstract_annotation_tagger_worker.sh"

METAGENOMICS_FULLTEXT_TAGGER_PATH="${LIB_PATH}/metagenomics_fulltext_annotation_tagger_worker.sh"




# Ensure necessary directories exist
mkdir -p "${TODAY_OUTPUT_DIR}/fulltext/sections" || {
    echo "Error: Failed to create output directories at ${TODAY_OUTPUT_DIR}/fulltext/sections"
    exit 1
}
mkdir -p "$FULLTEXT_LOG_DIR" || {
    echo "Error: Failed to create log directory at $FULLTEXT_LOG_DIR"
    exit 1
}
mkdir -p "$ABSTRACT_LOG_DIR" || {
    echo "Error: Failed to create log directory at $ABSTRACT_LOG_DIR"
    exit 1
}
mkdir -p "${TODAY_OUTPUT_DIR}/fulltext/annotations/europepmc" || {
    echo "Error: Failed to create output directories at ${TODAY_OUTPUT_DIR}/fulltext/annotations/europepmc"
    exit 1
}
mkdir -p "${TODAY_OUTPUT_DIR}/fulltext/annotations/metagenomics" || {
    echo "Error: Failed to create output directories at ${TODAY_OUTPUT_DIR}/fulltext/annotations/metagenomics"$
    exit 1
}
mkdir -p "${TODAY_OUTPUT_DIR}/abstract/annotations/europepmc" || {
    echo "Error: Failed to create directory at ${TODAY_OUTPUT_DIR}/abstract/annotations/europepmc"
    exit 1
}

# RUN THE ABSTRACT PIPELINE

# Count the number of abstract files to process
ABSTRACT_FILES=($(find "$ABSTRACT_SOURCE_DIR" -type f -name "*.abstract.gz" | sort))
NUM_ABSTRACT_FILES=${#ABSTRACT_FILES[@]}

# Submit abstract annotation tagger job array if files are found
if [[ $NUM_ABSTRACT_FILES -gt 0 ]]; then
    echo "Found $NUM_ABSTRACT_FILES abstract files to process."

    # Submit the abstract annotation tagger job array
    sbatch --array=0-$((NUM_ABSTRACT_FILES - 1)) \
           --job-name="AB-AN-${TIMESTAMP}" \
           --output="${ABSTRACT_LOG_DIR}/AB-AN_%A_%a.out" \
           --error="${ABSTRACT_LOG_DIR}/AB-AN_%A_%a.err" \
           --ntasks=1 \
           --cpus-per-task=3 \
           --partition=production \
           --mem=8G \
           --time=10:00:00 \
           --mail-user="stirunag@ebi.ac.uk" \
           --mail-type=BEGIN,END,FAIL,ARRAY_TASKS \
           "$ABSTRACT_TAGGER_PATH" "$ABSTRACT_SOURCE_DIR" "$TODAY_OUTPUT_DIR" "$MODEL_PATH_QUANTIZED"

    echo "Abstract annotation tagger job array submitted."
else
    echo "No abstract files found to process in $ABSTRACT_SOURCE_DIR"
fi


#RUN THE FULLTEXT PIPELINE
# Count the number of full-text files to process
FULLTEXT_FILES=($(find "$FULLTEXT_SOURCE_DIR" -type f -name "patch-*.xml.gz" | sort))
NUM_FULLTEXT_FILES=${#FULLTEXT_FILES[@]}

#1. SECTION TAGGER
# Count the number of files to process
#NUM_FILES=$(find "$FULLTEXT_SOURCE_DIR" -type f -name "patch-*.xml.gz" | wc -l)

# Check if there are any files to process
if [[ $NUM_FULLTEXT_FILES -gt 0 ]]; then
    echo "Found $NUM_FULLTEXT_FILES files to process."

    # Submit the section_tagger job and capture its job ID
    SECTION_JOB_ID=$(sbatch --array=0-$((NUM_FULLTEXT_FILES - 1)) \
                   --job-name="FT-ST-${TIMESTAMP}" \
                   --output="${FULLTEXT_LOG_DIR}/FT-ST_%A_%a.out" \
                   --error="${FULLTEXT_LOG_DIR}/FT-ST_%A_%a.err" \
                   --ntasks=1 \
                   --cpus-per-task=3 \
                   --nodes=1 \
                   --partition=production \
                   --mem=2G \
                   --time=10:00:00 \
                   --mail-user="stirunag@ebi.ac.uk" \
                   --mail-type=BEGIN,END,FAIL,ARRAY_TASKS \
                   "$SECTION_TAGGER_PATH" "$FULLTEXT_SOURCE_DIR" "$TODAY_OUTPUT_DIR" "$YESTERDAY_OUTPUT" \
                   | awk '{print $4}')

    echo "Section tagger job array submitted with Job ID: $SECTION_JOB_ID"

    # Submit annotation_tagger job array with dependency on section_tagger job completion
    sbatch --dependency=afterok:${SECTION_JOB_ID} \
           --array=0-$((NUM_FULLTEXT_FILES - 1)) \
           --job-name="FT-AN-${TIMESTAMP}" \
           --output="${FULLTEXT_LOG_DIR}/FT-AN_%A_%a.out" \
           --error="${FULLTEXT_LOG_DIR}/FT-AN_%A_%a.err" \
           --ntasks=1 \
           --cpus-per-task=3 \
           --nodes=1 \
           --partition=production \
           --mem=8G \
           --time=10:00:00 \
           --mail-user="stirunag@ebi.ac.uk" \
           --mail-type=BEGIN,END,FAIL,ARRAY_TASKS \
           "$FULLTEXT_TAGGER_PATH" "$TODAY_OUTPUT_DIR" "$MODEL_PATH_QUANTIZED"
    
    echo "Fulltext Annotation tagger job array submitted with dependency on Section tagger job ID: $SECTION_JOB_ID"

    # Submit metagenomics_fulltext_annotation_tagger job array with dependency on section_tagger job completion
    sbatch --dependency=afterok:${SECTION_JOB_ID} \
         --array=0-$((NUM_FULLTEXT_FILES - 1)) \
         --job-name="MTFT-AN-${TIMESTAMP}" \
         --output="${FULLTEXT_LOG_DIR}/MTFT-AN_%A_%a.out" \
         --error="${FULLTEXT_LOG_DIR}/MTFT-AN_%A_%a.err" \
         --ntasks=1 \
         --cpus-per-task=3 \
         --nodes=1 \
         --partition=production \
         --mem=8G \
         --time=10:00:00 \
         --mail-user="stirunag@ebi.ac.uk" \
         --mail-type=BEGIN,END,FAIL,ARRAY_TASKS \
         "$METAGENOMICS_FULLTEXT_TAGGER_PATH" "$TODAY_OUTPUT_DIR"

    echo "Metagenomics Fulltext Annotation tagger job array submitted with dependency on Section tagger job ID: $SECTION_JOB_ID"

else
    echo "No valid files found to process in $FULLTEXT_SOURCE_DIR."
    exit 1
fi

