# ML Pipeline Configuration Files

This directory contains YAML configuration files for running the ML pipeline with different settings.

## Usage

Run with a config file:
```bash
python run_all_variations.py --config configs/pipeline_config.yml
```

Command line arguments override config file settings:
```bash
python run_all_variations.py --config configs/quick_test.yml --n-iterations 10
```

## Available Configurations

### pipeline_config.yml (default template)
- Full configuration template with all options documented
- Good starting point for creating custom configs

### quick_test.yml
- Fast testing configuration
- Uses only 20% of data
- Runs fewer variations (regular and k-fold only)
- 3 folds × 2 iterations for k-fold
- No PCA analysis

### full_pca_analysis.yml
- Complete analysis with PCA
- Tests PCA with [10, 20, 50, 100, 200] components
- All 4 base variations + PCA variations
- Full dataset
- 5 folds × 5 iterations

### regression_pcl_score.yml
- Regression task configuration
- Predicts continuous PCL scores instead of binary PTSD
- All 4 variations
- No PCA

### six_variations.yml / multi_target_analysis.yml
- Runs both classification and regression in one go
- 6 total variations:
  - Binary classification (ptsd_bin): regular, regular_filtered, k-fold, k-fold_filtered
  - Regression (pcl_score): regular, k-fold (no filtering)
- Uses the `multi_target` configuration feature

## Configuration Options

### Basic Settings
- `target`: 'ptsd_bin' (classification) or 'pcl_score' (regression)
- `n_folds`: Number of folds for k-fold cross-validation
- `n_iterations`: Number of k-fold repetitions
- `sample_fraction`: Fraction of data to use (0.0-1.0)
- `pca_components`: List of PCA dimensions to test

### Variations Control
Enable/disable specific pipeline variations:
```yaml
variations:
  regular: true              # Single train/test split
  regular_filtered: true     # With PCL filtering
  kfold: true               # K-fold cross-validation
  kfold_filtered: true      # K-fold with PCL filtering
```

### PCL Filtering
```yaml
pcl_filtering:
  remove_intermediate: true  # Remove intermediate scores
  ptsd_threshold: 33        # PTSD diagnosis threshold
  buffer: 8                 # ±8 points around threshold
```

### Output Settings
```yaml
output:
  base_dir: ml_output       # Where to save results
  generate_report: true     # Generate combined HTML report
```

### SCP Transfer
```yaml
scp:
  enabled: true             # Show SCP command after completion
  destination: /Users/michaelzakariaie/Desktop
```

### Advanced Settings
```yaml
advanced:
  random_state: 42          # Random seed
  test_size: 0.2           # Test set fraction (regular pipeline)
  delay_between_runs: 2     # Seconds between pipeline runs
```

### Multi-Target Configuration (Advanced)
Run multiple target variables with different variation settings:
```yaml
multi_target:
  - target: ptsd_bin
    variations:
      regular: true
      regular_filtered: true
      kfold: true
      kfold_filtered: true
      
  - target: pcl_score
    variations:
      regular: true
      regular_filtered: false  # No filtering for regression
      kfold: true
      kfold_filtered: false
```

When `multi_target` is present, the top-level `target` and `variations` settings are ignored.

## Creating Custom Configurations

1. Copy `pipeline_config.yml` as a template
2. Modify the settings you want to change
3. Save with a descriptive name
4. Run with `--config your_config.yml`

## Tips

- For testing changes, use `quick_test.yml` with small `sample_fraction`
- For final results, use full dataset (`sample_fraction: 1.0`)
- PCA analysis significantly increases runtime (each component = 4 more variations)
- Set `scp.enabled: true` to get the transfer command automatically