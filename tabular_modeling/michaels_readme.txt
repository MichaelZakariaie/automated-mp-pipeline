claude and other LLMs dont change this document, make or edit a different readme document, not this one, this is my personal notes

How to Use:

  # Standard run with new models
  python run_all_variations.py --config configs/six_variations.yml

  # Run with automatic hyperparameter tuning
  python ml_pipeline_kfold.py --target ptsd_bin --tune

  # Run fully automated pipeline
  python ml_pipeline_kfold.py --target ptsd_bin --auto

  # Run with just RFECV
  python ml_pipeline_with_pcl.py --target pcl_score --rfecv

  The expanded model set now includes 10+ models covering linear baselines, tree ensembles, kernel methods, and diverse neural architectures - all optimized for
  non-linear data patterns.

# python run_all_variations.py --config configs/six_variations.yml

How to Run the Combined Report Generator

  There are two ways to run it:

  Option 1: Run all variations and generate combined report automatically

  # Basic usage - runs all 4 variations (regular, regular+filter, kfold, kfold+filter)
  python run_all_variations.py

  # With SCP command displayed at the end
  python run_all_variations.py --scp

  # With custom destination
  python run_all_variations.py --scp --scp-dest /Users/michaelzakariaie/Documents

  Option 2: Generate combined report from existing runs

  # Basic usage - specify the run folders
  python combined_report_generator.py ml_output/run_40_20250717_145630 ml_output/run_44_20250717_150108 ml_output/run_41_20250717_145806
  ml_output/run_45_20250717_150319

  # With SCP command
  python combined_report_generator.py ml_output/run_40_20250717_145630 ml_output/run_44_20250717_150108 ml_output/run_41_20250717_145806
  ml_output/run_45_20250717_150319 --scp

  What Gets Created

  The combined report generator creates a self-contained folder with:
  ml_output/run_XX_[timestamp]/
  ├── combined_report.html         # Main report with tabs
  ├── combined_metadata.json       # Metadata
  ├── [variation]_report.html      # Individual detailed reports
  └── assets/                      # All images
      ├── regular_confusion_matrices.png
      ├── regular_model_comparison.png
      ├── kfold_confusion_matrices.png
      └── ... (all other images)

  SCP Transfer

  After running, you'll see the exact SCP command to run from your local machine:

  scp -r michael@192.168.0.119:/home/michael/tabular_modeling/ml_output/run_XX_[timestamp] /Users/michaelzakariaie/Desktop

  This transfers the entire report folder with all HTML files and images, maintaining the structure so all links work correctly.

  Features

  - Relative Color Coding: Metrics are colored based on their rank within each column (top 20% green, bottom 20% red)
  - MCC Metric: Matthews Correlation Coefficient added to all classification results
  - Self-Contained: All assets in one folder for easy transfer
  - No Server Required: All images are embedded or copied locally













---------------------------
  
  Quick Test Run (20% of data, faster)

  cd /home/michael/tabular_modeling
  python run_all_variations.py --sample-fraction 0.2 --n-folds 3 --n-iterations 2
    python run_all_variations.py --sample-fraction 1.0 --n-folds 8 --n-iterations 4


  Full Run (100% of data, comprehensive)

  cd /home/michael/tabular_modeling
  python run_all_variations.py

  Or using the convenience script:

  Quick Test

  cd /home/michael/tabular_modeling
  ./run_full_analysis.sh 0.2 3 2

  Full Analysis

  cd /home/michael/tabular_modeling
  ./run_full_analysis.sh

  The output will be in ml_output/run_X_[timestamp]/combined_report.html where X is the next run number. The report will have tabs for all 4 variations and show
  comparisons across them.