#!/bin/bash
# Simple script to run time series analysis using multi-environment setup

echo "Starting Automated MP Pipeline - Time Series Analysis (Multi-Environment)"
echo "========================================================================"
echo ""

# Check if environments are set up
if [ ! -d "env_main" ] || [ ! -d "env_yochlol" ] || [ ! -d "env_cv" ]; then
    echo "ERROR: Virtual environments not found."
    echo "Please run './setup_multi_env.sh' first to set up the environments."
    exit 1
fi

# Activate main environment and run
echo "Activating main environment and running time series stage..."
source env_main/bin/activate

python multi_env_pipeline.py --stage time_series "$@"

deactivate

echo ""
echo "========================================================================"
echo "Time series analysis complete!"
echo ""
echo "Next steps:"
echo "1. Wait for external team to process data"
echo "2. Run ./run_tabular_multi.sh to complete analysis"