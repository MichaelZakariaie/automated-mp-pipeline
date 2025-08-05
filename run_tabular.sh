#!/bin/bash
# Simple script to run tabular analysis (second command)

echo "Starting Automated MP Pipeline - Tabular Analysis"
echo "================================================="
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run the tabular pipeline
python run_pipeline.py tabular "$@"

echo ""
echo "================================================="
echo "Pipeline complete! Check the reports directory for results."