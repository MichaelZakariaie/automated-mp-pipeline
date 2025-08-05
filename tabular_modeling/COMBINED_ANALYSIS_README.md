# Combined ML Pipeline Analysis

## Overview
Run all 4 ML pipeline variations with a single command and get a combined HTML report with tabs for easy comparison.

## The 4 Variations
1. **Regular** - Single train/test split
2. **Regular + PCL Filter** - Single split with intermediate PCL scores (25-41) removed
3. **K-Fold** - Cross-validation with multiple folds
4. **K-Fold + PCL Filter** - Cross-validation with PCL filtering

## Quick Start

### Full Analysis (All Data)
```bash
./run_full_analysis.sh
```

### Fast Test (20% of Data)
```bash
./run_full_analysis.sh 0.2 3 2
# Uses 20% sample, 3 folds, 2 iterations
```

### Direct Command
```bash
python run_all_variations.py --sample-fraction 0.2 --n-folds 3 --n-iterations 2
```

## What's New

### 1. Enhanced Metrics
- Added **Precision**, **Recall**, and **F1 Score** to all reports
- Better understanding of model performance beyond just accuracy

### 2. Data Sampling
- `--sample-fraction` parameter for faster testing
- Stratified sampling preserves class balance

### 3. Combined Report Features
- **Tabbed Interface** - Switch between variations easily
- **Overall Comparison Plot** - See all models across all variations
- **Key Insights** - Automatic analysis of results
- **Performance Warnings** - Alerts when models perform poorly

### 4. Single Command Execution
- Runs all 4 variations automatically
- Generates combined report with all results
- Each variation gets its own run folder plus a combined report folder

## Output Structure
```
ml_output/
├── run_8_20250717_131714/      # Regular
├── run_9_20250717_131736/      # Regular + PCL Filter  
├── run_10_20250717_131802/     # K-Fold
├── run_11_20250717_131826/     # K-Fold + PCL Filter
└── run_12_20250717_131834/     # Combined Report
    ├── combined_report.html     # Main report with tabs
    └── combined_metadata.json   # Metadata about all runs
```

## Understanding Your Results

### Why Performance is Poor
Your results show AUC ~0.5 (random chance) because:
1. Eye-tracking features may not strongly correlate with PTSD
2. Sample size might be too small
3. Feature engineering may need improvement
4. PTSD is complex and may require multimodal data

### Precision vs Recall
- **Precision**: Of all positive predictions, how many were correct?
- **Recall**: Of all actual positives, how many did we find?
- With imbalanced classes, precision can look decent even when the model is poor

### PCL Filtering Impact
Removing intermediate scores (25-41):
- Reduces samples from 481 → 303
- Creates clearer separation between classes
- Shows modest improvement in some models (XGBoost: +3.3% AUC)

## Tips
1. Use `--sample-fraction 0.1` for quick testing during development
2. Check individual run reports for detailed visualizations
3. The combined report shows relative performance across all variations
4. Port forward 8888 in VS Code to view reports easily