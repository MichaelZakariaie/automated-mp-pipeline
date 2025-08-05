# YAML Configuration Usage for ML Pipeline

## Quick Start with YAML Config

Instead of typing long commands with many arguments, you can now use YAML configuration files:

```bash
# Quick test run
python run_all_variations.py --config configs/quick_test.yml

# Full analysis with PCA
python run_all_variations.py --config configs/full_pca_analysis.yml

# Regression task
python run_all_variations.py --config configs/regression_pcl_score.yml

# Default configuration
python run_all_variations.py --config configs/pipeline_config.yml
```

## Override Config Settings

Command line arguments override YAML settings:

```bash
# Use quick test config but with more iterations
python run_all_variations.py --config configs/quick_test.yml --n-iterations 5

# Use full config but enable SCP
python run_all_variations.py --config configs/pipeline_config.yml --scp
```

## Creating Your Own Config

1. Copy an existing config:
```bash
cp configs/pipeline_config.yml configs/my_config.yml
```

2. Edit the settings you want
3. Run with your config:
```bash
python run_all_variations.py --config configs/my_config.yml
```

## Common Scenarios

### Testing a Change Quickly
```bash
python run_all_variations.py --config configs/quick_test.yml
```
- Uses 20% of data
- Runs only 2 variations
- 3 folds × 2 iterations
- Takes ~5-10 minutes

### Full Analysis
```bash
python run_all_variations.py --config configs/pipeline_config.yml
```
- All 4 variations
- Full dataset
- 5 folds × 5 iterations
- Takes ~30-60 minutes

### PCA Component Analysis
```bash
python run_all_variations.py --config configs/full_pca_analysis.yml
```
- Tests [10, 20, 50, 100, 200] PCA components
- 24 total variations (4 base + 20 PCA)
- Takes several hours

### With Automatic SCP Command
```bash
python run_all_variations.py --config configs/pipeline_config.yml --scp
```
Shows the exact SCP command to transfer results after completion

## Key Config Options

- `target`: 'ptsd_bin' or 'pcl_score'
- `sample_fraction`: 0.0-1.0 (portion of data to use)
- `n_folds`: Number of k-fold splits
- `n_iterations`: K-fold repetitions
- `pca_components`: List of PCA dimensions
- `variations`: Enable/disable specific pipeline types
- `scp.enabled`: Show SCP transfer command

See `configs/README.md` for full documentation of all options.