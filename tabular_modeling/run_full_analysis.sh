#!/bin/bash
# Run full ML analysis with all variations

echo "Running full ML pipeline analysis..."
echo "===================================="

# Default to full data, but allow override
SAMPLE_FRACTION=${1:-1.0}
FOLDS=${2:-5}
ITERATIONS=${3:-5}
# Optional: pass PCA components as 4th argument
PCA_COMPONENTS=${4:-""}

if [ "$SAMPLE_FRACTION" != "1.0" ]; then
    echo "Using ${SAMPLE_FRACTION} fraction of data for faster testing"
fi

echo "K-fold settings: $FOLDS folds, $ITERATIONS iterations"

if [ -n "$PCA_COMPONENTS" ]; then
    echo "PCA components to test: $PCA_COMPONENTS"
fi

echo ""

# Build command
CMD="python run_all_variations.py --sample-fraction $SAMPLE_FRACTION --n-folds $FOLDS --n-iterations $ITERATIONS"

# Add PCA components if provided
if [ -n "$PCA_COMPONENTS" ]; then
    CMD="$CMD --pca-components $PCA_COMPONENTS"
fi

# Run the combined pipeline
eval $CMD

echo ""
echo "Analysis complete! Check the ml_output folder for the combined report."