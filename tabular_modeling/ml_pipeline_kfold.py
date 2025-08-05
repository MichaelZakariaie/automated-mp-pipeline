#!/usr/bin/env python3
"""
K-Fold Cross-Validation ML Pipeline for PCL/PTSD prediction
Performs multiple iterations of k-fold validation and reports averaged metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import (
    make_scorer, accuracy_score, roc_auc_score, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, roc_curve,
    matthews_corrcoef
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import json
from collections import defaultdict

class KFoldPCLPipeline:
    """K-Fold Cross-Validation Pipeline for PCL/PTSD prediction"""
    
    def __init__(self, data_path, target_column='ptsd_bin', n_splits=5, n_iterations=5, random_state=42,
                 remove_intermediate_pcl=False, ptsd_threshold=33, buffer=8, sample_fraction=1.0,
                 use_pca=False, n_components=None, include_models=None, exclude_models=None):
        """
        Initialize the k-fold pipeline
        
        Args:
            data_path: Path to the processed data file with PCL scores
            target_column: Column to use as target variable ('ptsd_bin' or 'pcl_score')
            n_splits: Number of folds for k-fold cross-validation
            n_iterations: Number of times to repeat k-fold with different random states
            random_state: Base random seed for reproducibility
            remove_intermediate_pcl: Whether to remove intermediate PCL scores
            ptsd_threshold: PCL threshold for PTSD diagnosis
            buffer: Buffer around threshold to remove
        """
        self.data_path = Path(data_path)
        self.target_column = target_column
        self.n_splits = n_splits
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.is_regression = target_column == 'pcl_score'
        self.remove_intermediate_pcl = remove_intermediate_pcl
        self.ptsd_threshold = ptsd_threshold
        self.buffer = buffer
        self.sample_fraction = sample_fraction
        self.use_pca = use_pca
        self.n_components = n_components
        self.include_models = include_models
        self.exclude_models = exclude_models if exclude_models else []
        self.models = {}
        self.results = defaultdict(lambda: defaultdict(list))
        self.aggregated_results = {}
        self.predictions_collection = defaultdict(lambda: {'y_true': [], 'y_pred': [], 'y_proba': []})
        
    def load_data(self):
        """Load and prepare the data"""
        print(f"Loading data from {self.data_path}...")
        
        if self.data_path.suffix == '.parquet':
            self.df = pd.read_parquet(self.data_path)
        else:
            self.df = pd.read_csv(self.data_path)
            
        print(f"Data shape: {self.df.shape}")
        
        # Filter out rows without target values
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=[self.target_column])
        removed_count = initial_count - len(self.df)
        
        if removed_count > 0:
            print(f"Removed {removed_count} rows without {self.target_column}")
        
        # Remove intermediate PCL scores if requested
        if self.remove_intermediate_pcl and 'pcl_score' in self.df.columns:
            before_removal = len(self.df)
            lower_bound = self.ptsd_threshold - self.buffer
            upper_bound = self.ptsd_threshold + self.buffer
            
            print(f"\nRemoving intermediate PCL scores ({lower_bound} to {upper_bound})...")
            self.df = self.df[
                ~self.df['pcl_score'].between(
                    lower_bound, upper_bound, inclusive='both'
                )
            ]
            
            removed_intermediate = before_removal - len(self.df)
            print(f"Removed {removed_intermediate} samples with intermediate PCL scores")
            
            if 'ptsd_bin' in self.df.columns:
                print("\nPTSD distribution after filtering:")
                ptsd_counts = self.df['ptsd_bin'].value_counts(dropna=False).sort_index()
                for idx, count in ptsd_counts.items():
                    label = 'Negative' if idx == 0 else 'Positive' if idx == 1 else 'Missing'
                    print(f"  {label}: {count}")
        
        # Sample data if requested (for faster testing)
        if self.sample_fraction < 1.0:
            before_sample = len(self.df)
            if self.is_regression:
                self.df = self.df.sample(frac=self.sample_fraction, random_state=self.random_state)
            else:
                # Stratified sampling for classification
                self.df = self.df.groupby(self.target_column).apply(
                    lambda x: x.sample(frac=self.sample_fraction, random_state=self.random_state)
                ).reset_index(drop=True)
            print(f"\nSampled {self.sample_fraction*100:.0f}% of data: {before_sample} → {len(self.df)} samples")
        
        # Remove identifier and target columns from features
        exclude_columns = [
            'session_id', 'source_file', 'pcl_score', 'ptsd', 'ptsd_bin', 'session_id_full'
        ]
        
        # Get numeric features only
        self.feature_columns = []
        for col in self.df.columns:
            if col not in exclude_columns:
                if self.df[col].dtype in ['int64', 'float64'] or pd.api.types.is_numeric_dtype(self.df[col]):
                    self.feature_columns.append(col)
                elif self.df[col].dtype == 'object':
                    # Try to convert categorical to numeric
                    if set(self.df[col].dropna().unique()).issubset({'good', 'bad'}):
                        self.df[col] = self.df[col].map({'good': 1, 'bad': 0})
                        self.feature_columns.append(col)
        
        self.X = self.df[self.feature_columns].values
        self.y = self.df[self.target_column].values
        
        print(f"Features: {len(self.feature_columns)} columns")
        print(f"Samples: {len(self.X)}")
        
        if self.is_regression:
            print(f"\nTarget (PCL score) statistics:")
            print(f"  Mean: {self.y.mean():.2f}")
            print(f"  Std: {self.y.std():.2f}")
            print(f"  Range: [{self.y.min()}, {self.y.max()}]")
        else:
            print(f"\nTarget (PTSD) distribution:")
            unique, counts = np.unique(self.y, return_counts=True)
            for val, count in zip(unique, counts):
                print(f"  Class {val}: {count} ({count/len(self.y)*100:.1f}%)")
    
    def create_models(self):
        """Create ML models based on task type"""
        print(f"\nCreating models for {'regression' if self.is_regression else 'classification'}...")
        
        # Define all available models
        if self.is_regression:
            self.models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'Gradient Boosting': GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=self.random_state
                ),
                'XGBoost': xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=self.random_state,
                    eval_metric='rmse'
                ),
                'SVM': SVR(kernel='rbf', gamma='scale', C=1.0),
                'SVM_Poly': SVR(kernel='poly', degree=3, gamma='scale', C=1.0),
                'Neural Network': MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    activation='relu',
                    solver='adam',
                    max_iter=500,
                    random_state=self.random_state
                ),
                'Deep NN': MLPRegressor(
                    hidden_layer_sizes=(200, 100, 50, 25),
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    learning_rate='adaptive',
                    max_iter=1000,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=self.random_state
                ),
                'Wide NN': MLPRegressor(
                    hidden_layer_sizes=(300,),
                    activation='tanh',
                    solver='lbfgs',
                    alpha=0.01,
                    max_iter=1000,
                    random_state=self.random_state
                ),
                'Funnel NN': MLPRegressor(
                    hidden_layer_sizes=(256, 128, 64, 32, 16),
                    activation='relu',
                    solver='adam',
                    batch_size=32,
                    learning_rate_init=0.001,
                    alpha=0.0001,
                    max_iter=1500,
                    early_stopping=True,
                    n_iter_no_change=20,
                    random_state=self.random_state
                )
            }
        else:
            self.models = {
                'Logistic Regression': LogisticRegression(
                    max_iter=1000,
                    random_state=self.random_state
                ),
                'Naive Bayes': GaussianNB(),
                'Random Forest': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.random_state,
                    n_jobs=-1,
                    class_weight='balanced'
                ),
                'Gradient Boosting': GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=self.random_state
                ),
                'XGBoost': xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=self.random_state,
                    eval_metric='logloss'
                ),
                'SVM': SVC(kernel='rbf', gamma='scale', C=1.0, probability=True, class_weight='balanced'),
                'SVM_Poly': SVC(kernel='poly', degree=3, gamma='scale', C=1.0, probability=True, class_weight='balanced'),
                'Neural Network': MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    activation='relu',
                    solver='adam',
                    max_iter=500,
                    random_state=self.random_state
                ),
                'Deep NN': MLPClassifier(
                    hidden_layer_sizes=(200, 100, 50, 25),
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    learning_rate='adaptive',
                    max_iter=1000,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=self.random_state
                ),
                'Wide NN': MLPClassifier(
                    hidden_layer_sizes=(300,),
                    activation='tanh',
                    solver='lbfgs',
                    alpha=0.01,
                    max_iter=1000,
                    random_state=self.random_state
                ),
                'Funnel NN': MLPClassifier(
                    hidden_layer_sizes=(256, 128, 64, 32, 16),
                    activation='relu',
                    solver='adam',
                    batch_size=32,
                    learning_rate_init=0.001,
                    alpha=0.0001,
                    max_iter=1500,
                    early_stopping=True,
                    n_iter_no_change=20,
                    random_state=self.random_state
                )
            }
        
        # Apply model filtering based on include/exclude lists
        all_models = self.models.copy()
        
        if self.include_models:
            # Only keep models that are in the include list
            self.models = {name: model for name, model in all_models.items() 
                          if name in self.include_models}
            print(f"Including models: {list(self.models.keys())}")
        elif self.exclude_models:
            # Remove models that are in the exclude list
            self.models = {name: model for name, model in all_models.items() 
                          if name not in self.exclude_models}
            print(f"Excluding: {self.exclude_models}")
            print(f"Using models: {list(self.models.keys())}")
        else:
            print(f"Using all {len(self.models)} models")
    
    def get_scoring_metrics(self):
        """Get appropriate scoring metrics based on task type"""
        if self.is_regression:
            return {
                'r2': make_scorer(r2_score),
                'neg_mse': make_scorer(mean_squared_error, greater_is_better=False),
                'neg_mae': make_scorer(mean_absolute_error, greater_is_better=False),
                'neg_rmse': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), 
                                       greater_is_better=False)
            }
        else:
            from sklearn.metrics import precision_score, recall_score, f1_score
            return {
                'accuracy': make_scorer(accuracy_score),
                'roc_auc': make_scorer(roc_auc_score, needs_proba=True),
                'average_precision': make_scorer(average_precision_score, needs_proba=True),
                'precision': make_scorer(precision_score),
                'recall': make_scorer(recall_score),
                'f1': make_scorer(f1_score),
                'mcc': make_scorer(matthews_corrcoef)
            }
    
    def run_kfold_validation(self):
        """Run k-fold cross-validation multiple times"""
        print(f"\n{'='*60}")
        print(f"Running {self.n_iterations} iterations of {self.n_splits}-fold cross-validation")
        print(f"{'='*60}")
        
        # Create preprocessing pipeline
        preprocessing_steps = [
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]
        
        # Add PCA if requested
        if self.use_pca and self.n_components:
            preprocessing_steps.append(('pca', PCA(n_components=self.n_components, random_state=self.random_state)))
            print(f"Using PCA with {self.n_components} components")
        
        preprocessor = Pipeline(preprocessing_steps)
        
        # Get scoring metrics
        scoring = self.get_scoring_metrics()
        
        # For each model
        for model_name, base_model in self.models.items():
            print(f"\n{model_name}:")
            print("-" * 40)
            
            # Store results for all iterations
            iteration_results = defaultdict(list)
            
            # For collecting predictions (only do this for one iteration to save memory)
            collect_predictions = True
            
            # Run multiple iterations with different random states
            for iteration in range(self.n_iterations):
                print(f"  Iteration {iteration + 1}/{self.n_iterations}...", end='', flush=True)
                
                # Create k-fold splitter with different random state for each iteration
                if self.is_regression:
                    kf = KFold(n_splits=self.n_splits, shuffle=True, 
                              random_state=self.random_state + iteration)
                else:
                    kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, 
                                        random_state=self.random_state + iteration)
                
                # Create pipeline with preprocessing and model
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', base_model)
                ])
                
                # If we need to collect predictions for visualization
                if collect_predictions and not self.is_regression and iteration == 0:
                    fold_predictions = []
                    fold_true = []
                    fold_proba = []
                    
                    for train_idx, test_idx in kf.split(self.X, self.y):
                        X_train, X_test = self.X[train_idx], self.X[test_idx]
                        y_train, y_test = self.y[train_idx], self.y[test_idx]
                        
                        # Fit and predict
                        pipeline.fit(X_train, y_train)
                        y_pred = pipeline.predict(X_test)
                        
                        # Get probabilities if available
                        if hasattr(pipeline, 'predict_proba'):
                            y_proba = pipeline.predict_proba(X_test)[:, 1]
                            fold_proba.extend(y_proba)
                        
                        fold_predictions.extend(y_pred)
                        fold_true.extend(y_test)
                    
                    # Store predictions for this model
                    self.predictions_collection[model_name]['y_true'] = fold_true
                    self.predictions_collection[model_name]['y_pred'] = fold_predictions
                    if fold_proba:
                        self.predictions_collection[model_name]['y_proba'] = fold_proba
                
                # Run cross-validation
                cv_results = cross_validate(
                    pipeline, self.X, self.y, 
                    cv=kf, 
                    scoring=scoring,
                    n_jobs=-1,
                    return_train_score=True
                )
                
                # Store results for this iteration
                for metric, values in cv_results.items():
                    if metric.startswith('test_') or metric.startswith('train_'):
                        iteration_results[metric].extend(values)
                
                print(" Done!")
            
            # Calculate aggregated statistics across all folds and iterations
            self.results[model_name] = self.calculate_aggregated_metrics(iteration_results)
            
            # Print summary for this model
            self.print_model_summary(model_name)
    
    def calculate_aggregated_metrics(self, iteration_results):
        """Calculate mean and std across all k-fold iterations"""
        aggregated = {}
        
        for metric, values in iteration_results.items():
            values_array = np.array(values)
            aggregated[metric] = {
                'mean': values_array.mean(),
                'std': values_array.std(),
                'min': values_array.min(),
                'max': values_array.max(),
                'all_values': values_array.tolist()
            }
        
        return aggregated
    
    def print_model_summary(self, model_name):
        """Print summary statistics for a model"""
        results = self.results[model_name]
        
        print(f"\n  Summary across {self.n_iterations * self.n_splits} folds:")
        
        if self.is_regression:
            # R² Score
            r2_results = results['test_r2']
            print(f"    R² Score: {r2_results['mean']:.4f} ± {r2_results['std']:.4f} "
                  f"[{r2_results['min']:.4f}, {r2_results['max']:.4f}]")
            
            # RMSE (convert from negative)
            rmse_results = results['test_neg_rmse']
            rmse_mean = -rmse_results['mean']
            rmse_std = rmse_results['std']
            print(f"    RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}")
            
            # MAE (convert from negative)
            mae_results = results['test_neg_mae']
            mae_mean = -mae_results['mean']
            mae_std = mae_results['std']
            print(f"    MAE: {mae_mean:.4f} ± {mae_std:.4f}")
        else:
            # Accuracy
            acc_results = results['test_accuracy']
            print(f"    Accuracy: {acc_results['mean']:.4f} ± {acc_results['std']:.4f} "
                  f"[{acc_results['min']:.4f}, {acc_results['max']:.4f}]")
            
            # AUC
            auc_results = results['test_roc_auc']
            print(f"    AUC: {auc_results['mean']:.4f} ± {auc_results['std']:.4f} "
                  f"[{auc_results['min']:.4f}, {auc_results['max']:.4f}]")
            
            # Average Precision
            ap_results = results['test_average_precision']
            print(f"    Avg Precision: {ap_results['mean']:.4f} ± {ap_results['std']:.4f}")
    
    def hyperparameter_tuning(self, model_names=None, use_random_search=True, n_iter=20):
        """Perform hyperparameter tuning for specified models using k-fold CV"""
        if model_names is None:
            # Default to tuning the best performing models
            model_names = ['XGBoost', 'Random Forest', 'Gradient Boosting']
        
        param_distributions = {
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 10],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.5, 0.7, 1.0],
                'gamma': [0, 0.1, 0.5]
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.5]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.6, 0.8, 1.0]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }
        }
        
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING WITH K-FOLD")
        print("="*60)
        
        tuned_models = {}
        
        for model_name in model_names:
            if model_name in self.models and model_name in param_distributions:
                print(f"\nTuning {model_name}...")
                
                base_model = self.models[model_name]
                param_dist = param_distributions[model_name]
                
                # Use appropriate CV strategy
                cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state) if not self.is_regression else KFold(n_splits=5, shuffle=True, random_state=self.random_state)
                
                # Use RandomizedSearchCV
                if use_random_search:
                    search = RandomizedSearchCV(
                        base_model,
                        param_dist,
                        n_iter=n_iter,
                        cv=cv_strategy,
                        scoring=self.get_scoring_metrics(),
                        refit=list(self.get_scoring_metrics().keys())[0],
                        n_jobs=-1,
                        verbose=1,
                        random_state=self.random_state
                    )
                else:
                    search = GridSearchCV(
                        base_model,
                        param_dist,
                        cv=cv_strategy,
                        scoring=self.get_scoring_metrics(),
                        refit=list(self.get_scoring_metrics().keys())[0],
                        n_jobs=-1,
                        verbose=1
                    )
                
                # Preprocess data if not already done
                if not hasattr(self, 'X_processed'):
                    # Create preprocessing pipeline
                    imputer = SimpleImputer(strategy='median')
                    scaler = StandardScaler()
                    
                    # Fit and transform
                    X_imputed = imputer.fit_transform(self.X)
                    self.X_processed = scaler.fit_transform(X_imputed)
                
                # Fit on full training data
                search.fit(self.X_processed, self.y)
                
                print(f"Best parameters: {search.best_params_}")
                print(f"Best CV score: {search.best_score_:.4f}")
                
                # Store tuned model
                tuned_models[f'{model_name}_Tuned'] = search.best_estimator_
        
        # Add tuned models to the model list
        self.models.update(tuned_models)
        
        # Re-run k-fold with tuned models
        print("\nRe-running k-fold evaluation with tuned models...")
        self.run_pipeline()
        
        return tuned_models
    
    def feature_selection_rfecv(self, estimator=None, step=1, min_features=5):
        """Perform RFECV feature selection with k-fold cross-validation"""
        print("\n" + "="*60)
        print("RFECV FEATURE SELECTION")
        print("="*60)
        
        # Use Random Forest as default
        if estimator is None:
            if self.is_regression:
                estimator = RandomForestRegressor(n_estimators=50, random_state=self.random_state, n_jobs=-1)
            else:
                estimator = RandomForestClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1)
        
        # Create appropriate CV strategy
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state) if not self.is_regression else KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # Create RFECV
        rfecv = RFECV(
            estimator=estimator,
            step=step,
            cv=cv_strategy,
            scoring='r2' if self.is_regression else 'f1',
            min_features_to_select=min_features,
            n_jobs=-1,
            verbose=1
        )
        
        # Preprocess data if not already done
        if not hasattr(self, 'X_processed'):
            # Create preprocessing pipeline
            imputer = SimpleImputer(strategy='median')
            scaler = StandardScaler()
            
            # Fit and transform
            X_imputed = imputer.fit_transform(self.X)
            self.X_processed = scaler.fit_transform(X_imputed)
        
        # Fit RFECV
        rfecv.fit(self.X_processed, self.y)
        
        print(f"\nOptimal number of features: {rfecv.n_features_}")
        print(f"Features selected: {np.sum(rfecv.support_)} out of {len(rfecv.support_)}")
        
        # Transform data
        self.X_processed_rfecv = rfecv.transform(self.X_processed)
        self.rfecv = rfecv
        
        # Get selected features
        selected_features = np.array(self.feature_columns)[rfecv.support_]
        print(f"\nTop selected features: {list(selected_features[:10])}...")
        
        return rfecv
    
    def save_results(self, output_dir='ml_output'):
        """Save k-fold results and visualizations"""
        # Get next run folder with persistent counter
        from run_counter import get_next_run_folder
        output_path, run_number = get_next_run_folder(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Store run number for metadata
        self.run_number = run_number
        
        print(f"\nSaving results to: {output_path}")
        
        # Save detailed results
        self.save_detailed_results(output_path)
        
        # Create visualizations
        self.create_visualizations(output_path)
        
        # Save metadata
        self.save_metadata(output_path)
        
        # Generate unified HTML report
        try:
            from report_generator import generate_report_for_run
            report_path = generate_report_for_run(output_path, use_unified=True)
            print(f"Report saved: {report_path}")
        except Exception as e:
            print(f"Warning: Could not generate HTML report: {e}")
    
    def save_detailed_results(self, output_path):
        """Save detailed k-fold results to CSV"""
        # Create summary dataframe
        summary_data = []
        
        for model_name, model_results in self.results.items():
            row = {'Model': model_name}
            
            if self.is_regression:
                row.update({
                    'R² Mean': model_results['test_r2']['mean'],
                    'R² Std': model_results['test_r2']['std'],
                    'RMSE Mean': -model_results['test_neg_rmse']['mean'],
                    'RMSE Std': model_results['test_neg_rmse']['std'],
                    'MAE Mean': -model_results['test_neg_mae']['mean'],
                    'MAE Std': model_results['test_neg_mae']['std'],
                })
            else:
                row.update({
                    'Accuracy Mean': model_results['test_accuracy']['mean'],
                    'Accuracy Std': model_results['test_accuracy']['std'],
                    'AUC Mean': model_results['test_roc_auc']['mean'],
                    'AUC Std': model_results['test_roc_auc']['std'],
                    'Precision Mean': model_results.get('test_precision', {}).get('mean', 0),
                    'Precision Std': model_results.get('test_precision', {}).get('std', 0),
                    'Recall Mean': model_results.get('test_recall', {}).get('mean', 0),
                    'Recall Std': model_results.get('test_recall', {}).get('std', 0),
                    'F1 Mean': model_results.get('test_f1', {}).get('mean', 0),
                    'F1 Std': model_results.get('test_f1', {}).get('std', 0),
                    'Avg Precision Mean': model_results['test_average_precision']['mean'],
                    'Avg Precision Std': model_results['test_average_precision']['std'],
                    'MCC Mean': model_results.get('test_mcc', {}).get('mean', 'N/A'),
                    'MCC Std': model_results.get('test_mcc', {}).get('std', 'N/A'),
                })
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path / 'kfold_results_summary.csv', index=False)
        
        # Save detailed fold-by-fold results
        detailed_results = []
        for model_name, model_results in self.results.items():
            for metric_name, metric_data in model_results.items():
                if 'all_values' in metric_data:
                    for fold_idx, value in enumerate(metric_data['all_values']):
                        detailed_results.append({
                            'Model': model_name,
                            'Metric': metric_name,
                            'Fold': fold_idx,
                            'Value': value
                        })
        
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv(output_path / 'kfold_detailed_results.csv', index=False)
    
    def create_visualizations(self, output_path):
        """Create visualizations for k-fold results"""
        # Model comparison plot
        plt.figure(figsize=(12, 8))
        
        models = list(self.results.keys())
        
        if self.is_regression:
            # R² scores with error bars
            means = [self.results[m]['test_r2']['mean'] for m in models]
            stds = [self.results[m]['test_r2']['std'] for m in models]
            
            plt.bar(models, means, yerr=stds, capsize=5, alpha=0.7)
            plt.ylabel('R² Score')
            plt.title(f'K-Fold Cross-Validation Results (PCL Score Prediction)\n'
                     f'{self.n_iterations} iterations × {self.n_splits} folds = '
                     f'{self.n_iterations * self.n_splits} total evaluations')
        else:
            # AUC scores with error bars
            means = [self.results[m]['test_roc_auc']['mean'] for m in models]
            stds = [self.results[m]['test_roc_auc']['std'] for m in models]
            
            plt.bar(models, means, yerr=stds, capsize=5, alpha=0.7)
            plt.ylabel('AUC Score')
            plt.title(f'K-Fold Cross-Validation Results (PTSD Prediction)\n'
                     f'{self.n_iterations} iterations × {self.n_splits} folds = '
                     f'{self.n_iterations * self.n_splits} total evaluations')
        
        plt.xlabel('Models')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'kfold_model_comparison.png', dpi=300)
        plt.close()
        
        # Box plots showing distribution of scores across folds
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, (model_name, model_results) in enumerate(self.results.items()):
            if idx < len(axes):
                ax = axes[idx]
                
                if self.is_regression:
                    # Box plot of R² scores
                    data = model_results['test_r2']['all_values']
                    metric_name = 'R² Score'
                else:
                    # Box plot of AUC scores
                    data = model_results['test_roc_auc']['all_values']
                    metric_name = 'AUC Score'
                
                ax.boxplot(data)
                ax.set_title(f'{model_name}\n{metric_name} Distribution')
                ax.set_ylabel(metric_name)
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(self.results), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Score Distribution Across {self.n_iterations * self.n_splits} Folds', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path / 'kfold_score_distributions.png', dpi=300)
        plt.close()
        
        # For classification tasks, create confusion matrices and ROC curves
        if not self.is_regression and self.predictions_collection:
            self.create_classification_visualizations(output_path)
    
    def create_classification_visualizations(self, output_path):
        """Create confusion matrices and ROC curves for classification tasks"""
        # Confusion matrices
        n_models = len(self.predictions_collection)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for idx, (model_name, predictions) in enumerate(self.predictions_collection.items()):
            if idx < len(axes) and predictions['y_true'] and predictions['y_pred']:
                ax = axes[idx]
                
                # Calculate confusion matrix
                cm = confusion_matrix(predictions['y_true'], predictions['y_pred'])
                
                # Plot confusion matrix
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=['Negative', 'Positive'],
                           yticklabels=['Negative', 'Positive'])
                ax.set_title(f'{model_name}\nAggregate Confusion Matrix')
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('K-Fold Aggregate Confusion Matrices', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path / 'kfold_confusion_matrices.png', dpi=300)
        plt.close()
        
        # ROC curves
        plt.figure(figsize=(10, 8))
        
        for model_name, predictions in self.predictions_collection.items():
            if predictions['y_proba']:
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(predictions['y_true'], predictions['y_proba'])
                auc_score = roc_auc_score(predictions['y_true'], predictions['y_proba'])
                
                # Plot ROC curve
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        
        # Add diagonal reference line
        plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('K-Fold Aggregate ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'kfold_roc_curves.png', dpi=300)
        plt.close()
    
    def save_metadata(self, output_path):
        """Save metadata about the k-fold run"""
        metadata = {
            'run_number': self.run_number,
            'run_timestamp': datetime.now().isoformat(),
            'pipeline_type': 'k-fold cross-validation',
            'target_variable': self.target_column,
            'task_type': 'regression' if self.is_regression else 'classification',
            'validation_settings': {
                'n_folds': self.n_splits,
                'n_iterations': self.n_iterations,
                'total_evaluations': self.n_iterations * self.n_splits,
                'base_random_state': self.random_state
            },
            'data_info': {
                'source_file': str(self.data_path),
                'total_samples': len(self.X),
                'total_features': len(self.feature_columns)
            },
            'pcl_filtering': {
                'remove_intermediate': self.remove_intermediate_pcl,
                'ptsd_threshold': self.ptsd_threshold,
                'buffer': self.buffer,
                'excluded_range': f"{self.ptsd_threshold - self.buffer} to {self.ptsd_threshold + self.buffer}" if self.remove_intermediate_pcl else None
            },
            'pca': {
                'use_pca': self.use_pca,
                'n_components': self.n_components
            },
            'model_results_summary': {}
        }
        
        # Add summary results for each model
        for model_name, model_results in self.results.items():
            if self.is_regression:
                metadata['model_results_summary'][model_name] = {
                    'r2_mean': model_results['test_r2']['mean'],
                    'r2_std': model_results['test_r2']['std'],
                    'rmse_mean': -model_results['test_neg_rmse']['mean'],
                    'rmse_std': model_results['test_neg_rmse']['std'],
                    'mae_mean': -model_results['test_neg_mae']['mean'],
                    'mae_std': model_results['test_neg_mae']['std']
                }
            else:
                metadata['model_results_summary'][model_name] = {
                    'accuracy_mean': model_results['test_accuracy']['mean'],
                    'accuracy_std': model_results['test_accuracy']['std'],
                    'auc_mean': model_results['test_roc_auc']['mean'],
                    'auc_std': model_results['test_roc_auc']['std'],
                    'avg_precision_mean': model_results['test_average_precision']['mean'],
                    'avg_precision_std': model_results['test_average_precision']['std']
                }
        
        with open(output_path / 'kfold_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def run_pipeline(self):
        """Run the complete k-fold pipeline"""
        print("=" * 60)
        print(f"K-FOLD CROSS-VALIDATION PIPELINE")
        print(f"Target: {self.target_column}")
        print(f"Task: {'Regression' if self.is_regression else 'Classification'}")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Create models
        self.create_models()
        
        # Run k-fold validation
        self.run_kfold_validation()
        
        # Save results
        self.save_results()
        
        print("\n" + "=" * 60)
        print("K-fold pipeline completed successfully!")
        print("=" * 60)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run k-fold cross-validation pipeline for PCL/PTSD prediction')
    parser.add_argument('--data-path', default='processed_sessions/all_sessions_with_pcl.parquet',
                        help='Path to processed data file with PCL scores')
    parser.add_argument('--target', default='ptsd_bin', choices=['ptsd_bin', 'pcl_score'],
                        help='Target column for prediction')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    parser.add_argument('--n-iterations', type=int, default=5,
                        help='Number of times to repeat k-fold validation')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--remove-intermediate-pcl', action='store_true',
                        help='Remove samples with PCL scores near the PTSD threshold')
    parser.add_argument('--ptsd-threshold', type=int, default=33,
                        help='PCL threshold for PTSD diagnosis')
    parser.add_argument('--buffer', type=int, default=8,
                        help='Buffer around threshold to remove')
    parser.add_argument('--sample-fraction', type=float, default=1.0,
                        help='Fraction of data to use (for faster testing)')
    parser.add_argument('--use-pca', action='store_true',
                        help='Use PCA for dimensionality reduction')
    parser.add_argument('--n-components', type=int, default=None,
                        help='Number of PCA components to keep')
    parser.add_argument('--tune', action='store_true',
                        help='Perform hyperparameter tuning')
    parser.add_argument('--rfecv', action='store_true',
                        help='Perform RFECV feature selection')
    parser.add_argument('--auto', action='store_true',
                        help='Run automated pipeline with RFECV and tuning')
    parser.add_argument('--include-models', nargs='+', default=None,
                        help='List of models to include')
    parser.add_argument('--exclude-models', nargs='+', default=None,
                        help='List of models to exclude')
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = KFoldPCLPipeline(
        data_path=args.data_path,
        target_column=args.target,
        n_splits=args.n_folds,
        n_iterations=args.n_iterations,
        random_state=args.random_state,
        remove_intermediate_pcl=args.remove_intermediate_pcl,
        ptsd_threshold=args.ptsd_threshold,
        buffer=args.buffer,
        sample_fraction=args.sample_fraction,
        use_pca=args.use_pca,
        n_components=args.n_components,
        include_models=args.include_models,
        exclude_models=args.exclude_models
    )
    
    # Run pipeline based on arguments
    if args.auto:
        # Automated pipeline
        print("Running automated k-fold pipeline with RFECV and hyperparameter tuning...")
        pipeline.load_data()
        pipeline.preprocess_data()
        pipeline.create_models()
        
        # First do RFECV
        pipeline.feature_selection_rfecv()
        
        # Then run k-fold with selected features
        original_X = pipeline.X_processed
        pipeline.X_processed = pipeline.X_processed_rfecv
        pipeline.run_kfold()
        
        # Finally do hyperparameter tuning
        pipeline.hyperparameter_tuning()
        
        # Restore original features
        pipeline.X_processed = original_X
        
        # Save results
        pipeline.save_results()
    else:
        # Standard pipeline
        pipeline.run_pipeline()
        
        # Optional RFECV
        if args.rfecv:
            pipeline.feature_selection_rfecv()
        
        # Optional hyperparameter tuning
        if args.tune:
            pipeline.hyperparameter_tuning()

if __name__ == "__main__":
    main()