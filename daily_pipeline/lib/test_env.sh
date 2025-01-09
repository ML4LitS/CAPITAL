#!/bin/bash

ENV_FILE="$(dirname "$0")/../.env_paths"

if [ -f "$ENV_FILE" ]; then
    # Try to source the environment file
    if source "$ENV_FILE"; then
        echo "Loaded environment file from $ENV_FILE"
    else
        echo "Error: Could not load environment file from $ENV_FILE"
        exit 1
    fi
else
    echo "Error: Environment file not found at $ENV_FILE"
    exit 1
fi

