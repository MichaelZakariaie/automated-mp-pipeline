#!/bin/bash
# Simple script to run tabular analysis using multi-environment setup

echo "Starting Automated MP Pipeline - Tabular Analysis (Multi-Environment)"
echo "===================================================================="
echo ""

# Check if environments are set up
if [ ! -d "env_main" ] || [ ! -d "env_tabular" ]; then
    echo "ERROR: Virtual environments not found."
    echo "Please run './setup_multi_env.sh' first to set up the environments."
    exit 1
fi

# Activate main environment and run
echo "Activating main environment and running tabular stage..."
source env_main/bin/activate

python multi_env_pipeline.py --stage tabular "$@"

deactivate

echo ""
echo "===================================================================="
echo "Pipeline complete! Check the reports directory for results."