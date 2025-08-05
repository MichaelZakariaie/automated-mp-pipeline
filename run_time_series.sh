#!/bin/bash
# Simple script to run time series analysis (first command)

echo "Starting Automated MP Pipeline - Time Series Analysis"
echo "====================================================="
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run the time series pipeline
python run_pipeline.py time_series "$@"

echo ""
echo "====================================================="
echo "Time series analysis complete!"
echo ""
echo "Next steps:"
echo "1. Wait for external team to process data"
echo "2. Run ./run_tabular.sh to complete analysis"