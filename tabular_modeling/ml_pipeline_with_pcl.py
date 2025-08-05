#!/usr/bin/env python3
"""
ML Pipeline for analyzing session data with PCL/PTSD prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score, matthews_corrcoef
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import json

class PCLPredictionPipeline:
    """ML Pipeline for PCL/PTSD prediction from session data"""
    
    def __init__(self, data_path, target_column='ptsd_bin', test_size=0.2, random_state=42,
                 remove_intermediate_pcl=False, ptsd_threshold=33, buffer=8, sample_fraction=1.0,
                 use_pca=False, n_components=None, include_models=None, exclude_models=None):
        """
        Initialize the ML pipeline
        
        Args:
            data_path: Path to the processed data file with PCL scores
            target_column: Column to use as target variable ('ptsd_bin' or 'pcl_score')
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            remove_intermediate_pcl: Whether to remove intermediate PCL scores
            ptsd_threshold: PCL threshold for PTSD diagnosis
            buffer: Buffer around threshold to remove
            sample_fraction: Fraction of data to use (for faster testing)
            use_pca: Whether to use PCA for dimensionality reduction
            n_components: Number of PCA components to keep
        """
        self.data_path = Path(data_path)
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.is_regression = target_column == 'pcl_score'
        self.remove_intermediate_pcl = remove_intermediate_pcl
        self.ptsd_threshold = ptsd_threshold
        self.buffer = buffer
        self.sample_fraction = sample_fraction
        self.use_pca = use_pca
        self.n_components = n_components
        self.include_models = include_models
        self.exclude_models = exclude_models if exclude_models else []
        
    def display_data_info(self):
        """Display detailed information about the dataset"""
        print("\n" + "=" * 50)
        print("DATASET INFORMATION")
        print("=" * 50)
        
        # Basic info
        print(f"\n1. BASIC INFORMATION:")
        print(f"   - Total samples: {len(self.df)}")
        print(f"   - Total features: {len(self.df.columns)}")
        
        # Target variable info
        print(f"\n2. TARGET VARIABLE: {self.target_column}")
        if self.target_column in self.df.columns:
            if self.is_regression:
                print(f"   - Type: Regression")
                print(f"   - Mean: {self.df[self.target_column].mean():.2f}")
                print(f"   - Std: {self.df[self.target_column].std():.2f}")
                print(f"   - Min: {self.df[self.target_column].min()}")
                print(f"   - Max: {self.df[self.target_column].max()}")
                print(f"   - Missing values: {self.df[self.target_column].isna().sum()}")
            else:
                print(f"   - Type: Classification")
                print(f"   - Class distribution:")
                value_counts = self.df[self.target_column].value_counts(dropna=False)
                for val, count in value_counts.items():
                    print(f"     {val}: {count}")
        
        # PTSD information
        print(f"\n3. PTSD INFORMATION:")
        if 'ptsd' in self.df.columns:
            print(f"   - PTSD categories:")
            ptsd_counts = self.df['ptsd'].value_counts(dropna=False)
            for val, count in ptsd_counts.items():
                print(f"     {val}: {count}")
        if 'ptsd_bin' in self.df.columns:
            print(f"   - PTSD binary distribution:")
            ptsd_counts = self.df['ptsd_bin'].value_counts(dropna=False)
            for val, count in ptsd_counts.items():
                label = "Positive" if val == 1 else "Negative" if val == 0 else "Missing"
                pct = count / len(self.df) * 100
                print(f"     {label}: {count} ({pct:.1f}%)")
        
        # Feature categories
        print(f"\n4. FEATURE CATEGORIES:")
        feature_types = {}
        for col in self.df.columns:
            if any(keyword in col for keyword in ['trial', 'session', 'percent', 'quality', 'latency']):
                if 'trial' in col and col.startswith('trial'):
                    trial_num = col.split('_')[0].replace('trial', '')
                    if trial_num.isdigit():
                        feature_type = col.split('_', 1)[1] if '_' in col else 'unknown'
                        if feature_type not in feature_types:
                            feature_types[feature_type] = 0
                        feature_types[feature_type] += 1
        
        for feat_type, count in sorted(feature_types.items()):
            print(f"   - {feat_type}: {count} trials")
        
        # Data source info
        print(f"\n5. DATA SOURCE:")
        print(f"   - Data file: {self.data_path.name}")
        print(f"   - S3 bucket: senseye-data-quality")
        print(f"   - S3 path: messy_prototyping_saturn_uploads/")
        
        # Cohort information (inferred from available data)
        print(f"\n6. COHORT INFORMATION:")
        print(f"   - Cohort identification: Not explicitly available in current dataset")
        print(f"   - Total participants: {len(self.df)}")
        if 'ptsd_bin' in self.df.columns:
            ptsd_pos = (self.df['ptsd_bin'] == 1).sum()
            ptsd_neg = (self.df['ptsd_bin'] == 0).sum()
            print(f"   - PTSD positive participants: {ptsd_pos}")
            print(f"   - PTSD negative participants: {ptsd_neg}")
        
        print("=" * 50 + "\n")
        
    def load_data(self):
        """Load and prepare the data"""
        print(f"Loading data from {self.data_path}...")
        
        if self.data_path.suffix == '.parquet':
            self.df = pd.read_parquet(self.data_path)
        else:
            self.df = pd.read_csv(self.data_path)
            
        print(f"Data shape: {self.df.shape}")
        
        # Display detailed data information
        self.display_data_info()
        
        # Filter out rows without PCL scores
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
                print(self.df['ptsd_bin'].value_counts(dropna=False).sort_index())
        
        # Sample data if requested (for faster testing)
        if self.sample_fraction < 1.0:
            before_sample = len(self.df)
            if self.is_regression:
                self.df = self.df.sample(frac=self.sample_fraction, random_state=self.random_state)
            else:
                # Stratified sampling for classification
                self.df = self.df.groupby('ptsd_bin').apply(
                    lambda x: x.sample(frac=self.sample_fraction, random_state=self.random_state)
                ).reset_index(drop=True)
            print(f"\nSampled {self.sample_fraction*100:.0f}% of data: {before_sample} → {len(self.df)} samples")
        
        # Remove identifier and PCL-related columns from features
        exclude_columns = [
            'session_id', 'source_file', 'pcl_score', 'ptsd', 'ptsd_bin', 'session_id_full'
        ]
        
        # Also exclude columns with string/categorical data for now
        self.feature_columns = []
        for col in self.df.columns:
            if col not in exclude_columns:
                # Check if column is numeric
                if self.df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    self.feature_columns.append(col)
                elif self.df[col].dtype == 'object':
                    # Try to convert good/bad to numeric
                    if set(self.df[col].dropna().unique()).issubset({'good', 'bad', ''}):
                        self.df[col] = self.df[col].map({'good': 1, 'bad': 0}).fillna(self.df[col])
                        if self.df[col].dtype in ['float64', 'int64']:
                            self.feature_columns.append(col)
                    else:
                        print(f"Excluding non-numeric column: {col}")
                else:
                    self.feature_columns.append(col)
        
        # Prepare features and target
        self.X = self.df[self.feature_columns]
        self.y = self.df[self.target_column]
        
        print(f"Features: {len(self.feature_columns)} columns")
        
        # Print detailed feature information
        print(f"\n7. INPUT FEATURES (X):")
        print(f"   - Total input columns: {len(self.feature_columns)}")
        print(f"   - Feature types included:")
        feature_summary = {}
        for col in self.feature_columns:
            if 'trial' in col and col.startswith('trial'):
                feature_type = col.split('_', 1)[1] if '_' in col else 'unknown'
                if feature_type not in feature_summary:
                    feature_summary[feature_type] = 0
                feature_summary[feature_type] += 1
        
        for feat_type, count in sorted(feature_summary.items()):
            print(f"     - {feat_type}: {count} trials")
        
        print(f"\n8. OUTPUT TARGET (Y): {self.target_column}")
        
        if self.is_regression:
            print(f"Target (PCL score) statistics:")
            print(f"  Mean: {self.y.mean():.2f}")
            print(f"  Std: {self.y.std():.2f}")
            print(f"  Range: [{self.y.min()}, {self.y.max()}]")
        else:
            print(f"Target (PTSD) distribution:")
            print(self.y.value_counts())
            print(f"Class balance: {self.y.mean():.2%} positive")
        
    def preprocess_data(self):
        """Preprocess the data"""
        print("\nPreprocessing data...")
        
        # Create preprocessing pipeline
        preprocessing_steps = [
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]
        
        # Add PCA if requested
        if self.use_pca and self.n_components:
            preprocessing_steps.append(('pca', PCA(n_components=self.n_components, random_state=self.random_state)))
            print(f"Using PCA with {self.n_components} components")
        
        self.preprocessor = Pipeline(preprocessing_steps)
        
        # Split data with stratification for classification
        if self.is_regression:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=self.test_size, random_state=self.random_state
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=self.test_size, random_state=self.random_state, 
                stratify=self.y
            )
        
        # Fit and transform
        self.X_train_processed = self.preprocessor.fit_transform(self.X_train)
        self.X_test_processed = self.preprocessor.transform(self.X_test)
        
        print(f"Training set: {self.X_train_processed.shape}")
        print(f"Test set: {self.X_test_processed.shape}")
        
    def create_models(self):
        """Create various ML models"""
        print("\nCreating models...")
        
        if self.is_regression:
            # Regression models for PCL score prediction
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.svm import SVR
            from sklearn.neural_network import MLPRegressor
            from sklearn.linear_model import LinearRegression
            
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
                    random_state=self.random_state
                ),
                
                'SVM': SVR(
                    kernel='rbf',
                    gamma='scale',
                    C=1.0
                ),
                
                'SVM_Poly': SVR(
                    kernel='poly',
                    degree=3,
                    gamma='scale',
                    C=1.0
                ),
                
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
            # Classification models for PTSD prediction
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
                    class_weight='balanced'  # Handle class imbalance
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
                    eval_metric='logloss',
                    scale_pos_weight=len(self.y_train[self.y_train==0]) / len(self.y_train[self.y_train==1])
                ),
                
                'SVM': SVC(
                    kernel='rbf',
                    gamma='scale',
                    C=1.0,
                    probability=True,
                    random_state=self.random_state,
                    class_weight='balanced'
                ),
                
                'SVM_Poly': SVC(
                    kernel='poly',
                    degree=3,
                    gamma='scale',
                    C=1.0,
                    probability=True,
                    random_state=self.random_state,
                    class_weight='balanced'
                ),
                
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
        
    def train_models(self):
        """Train all models"""
        print("\nTraining models...")
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(self.X_train_processed, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test_processed)
            
            if self.is_regression:
                # Regression metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                train_score = model.score(self.X_train_processed, self.y_train)
                test_score = model.score(self.X_test_processed, self.y_test)
                mse = mean_squared_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(model, self.X_train_processed, self.y_train, cv=5)
                
                self.results[name] = {
                    'model': model,
                    'train_score': train_score,
                    'test_score': test_score,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'cv_scores': cv_scores,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_pred': y_pred
                }
                
                print(f"  Train R²: {train_score:.4f}")
                print(f"  Test R²: {test_score:.4f}")
                print(f"  MSE: {mse:.4f}")
                print(f"  MAE: {mae:.4f}")
                
            else:
                # Classification metrics
                y_pred_proba = model.predict_proba(self.X_test_processed)[:, 1]
                
                train_score = model.score(self.X_train_processed, self.y_train)
                test_score = model.score(self.X_test_processed, self.y_test)
                
                # Stratified cross-validation
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
                cv_scores = cross_val_score(model, self.X_train_processed, self.y_train, cv=cv)
                
                # Calculate metrics
                from sklearn.metrics import precision_score, recall_score, f1_score
                auc_score = roc_auc_score(self.y_test, y_pred_proba)
                ap_score = average_precision_score(self.y_test, y_pred_proba)
                precision = precision_score(self.y_test, y_pred)
                recall = recall_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred)
                
                self.results[name] = {
                    'model': model,
                    'train_score': train_score,
                    'test_score': test_score,
                    'cv_scores': cv_scores,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'classification_report': classification_report(self.y_test, y_pred),
                    'confusion_matrix': confusion_matrix(self.y_test, y_pred),
                    'mcc': matthews_corrcoef(self.y_test, y_pred),
                    'auc_score': auc_score,
                    'ap_score': ap_score,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
                
                print(f"  Train accuracy: {train_score:.4f}")
                print(f"  Test accuracy: {test_score:.4f}")
                print(f"  AUC: {auc_score:.4f}")
                print(f"  Average Precision: {ap_score:.4f}")
                print(f"  CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
    def feature_importance_analysis(self, output_path=None):
        """Analyze feature importance for tree-based models"""
        print("\nAnalyzing feature importance...")
        
        # Get feature importance from Random Forest
        rf_model = self.results['Random Forest']['model']
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot top 20 features
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(20)
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('Importance')
        plt.title(f'Top 20 Most Important Features for {self.target_column} Prediction')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path / 'feature_importance.png', dpi=300)
        plt.close()
        
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))
        
        return feature_importance
        
    def plot_roc_curves(self, output_path):
        """Plot ROC curves for all models (classification only)"""
        if self.is_regression:
            return
            
        plt.figure(figsize=(10, 8))
        
        for name, result in self.results.items():
            if 'y_pred_proba' in result:
                fpr, tpr, _ = roc_curve(self.y_test, result['y_pred_proba'])
                auc = result['auc_score']
                plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for PTSD Prediction')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'roc_curves.png', dpi=300)
        plt.close()
    
    def hyperparameter_tuning(self, model_names=None, use_random_search=True, n_iter=30):
        """Perform hyperparameter tuning for specified models"""
        if model_names is None:
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
                'gamma': ['scale', 'auto', 0.001, 0.01]
            }
        }
        
        tuned_models = {}
        scoring = 'r2' if self.is_regression else 'f1'
        
        for model_name in model_names:
            if model_name in self.models and model_name in param_distributions:
                print(f"\nTuning {model_name}...")
                
                base_model = self.models[model_name]
                param_dist = param_distributions[model_name]
                
                if use_random_search:
                    search = RandomizedSearchCV(
                        base_model,
                        param_dist,
                        n_iter=n_iter,
                        cv=5,
                        scoring=scoring,
                        n_jobs=-1,
                        verbose=1,
                        random_state=self.random_state
                    )
                else:
                    search = GridSearchCV(
                        base_model,
                        param_dist,
                        cv=5,
                        scoring=scoring,
                        n_jobs=-1,
                        verbose=1
                    )
                
                search.fit(self.X_train_processed, self.y_train)
                
                print(f"Best parameters: {search.best_params_}")
                print(f"Best CV score: {search.best_score_:.4f}")
                
                tuned_models[f'{model_name}_Tuned'] = search.best_estimator_
                
                test_score = search.score(self.X_test_processed, self.y_test)
                print(f"Test score: {test_score:.4f}")
        
        self.models.update(tuned_models)
        self.train_models()
        
        return tuned_models
    
    def feature_selection_rfecv(self, estimator=None, step=1, min_features=5):
        """Perform RFECV feature selection"""
        print("\nRunning RFECV Feature Selection...")
        
        if estimator is None:
            if self.is_regression:
                from sklearn.ensemble import RandomForestRegressor
                estimator = RandomForestRegressor(n_estimators=50, random_state=self.random_state, n_jobs=-1)
            else:
                from sklearn.ensemble import RandomForestClassifier
                estimator = RandomForestClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1)
        
        rfecv = RFECV(
            estimator=estimator,
            step=step,
            cv=5,
            scoring='r2' if self.is_regression else 'f1',
            min_features_to_select=min_features,
            n_jobs=-1,
            verbose=1
        )
        
        rfecv.fit(self.X_train_processed, self.y_train)
        
        print(f"\nOptimal features: {rfecv.n_features_}")
        print(f"Selected: {np.sum(rfecv.support_)} out of {len(rfecv.support_)}")
        
        self.X_train_processed_rfecv = rfecv.transform(self.X_train_processed)
        self.X_test_processed_rfecv = rfecv.transform(self.X_test_processed)
        self.rfecv = rfecv
        
        selected_features = np.array(self.feature_columns)[rfecv.support_]
        print(f"\nTop features: {list(selected_features[:10])}...")
        
        return rfecv
        
    def save_results(self, output_dir='ml_output'):
        """Save models and results"""
        # Get next run folder with persistent counter
        from run_counter import get_next_run_folder
        output_path, run_number = get_next_run_folder(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Store run number for metadata
        self.run_number = run_number
        
        print(f"\nSaving results to: {output_path}")
        
        # Save models
        for name, result in self.results.items():
            model_path = output_path / f"{name.lower().replace(' ', '_')}_model.pkl"
            joblib.dump(result['model'], model_path)
            print(f"Saved {name} model to {model_path}")
        
        # Save preprocessor
        preprocessor_path = output_path / "preprocessor.pkl"
        joblib.dump(self.preprocessor, preprocessor_path)
        print(f"Saved preprocessor to {preprocessor_path}")
        
        # Save results summary
        if self.is_regression:
            summary = []
            for name, result in self.results.items():
                summary.append({
                    'Model': name,
                    'Train R²': result['train_score'],
                    'Test R²': result['test_score'],
                    'MSE': result['mse'],
                    'RMSE': result['rmse'],
                    'MAE': result['mae'],
                    'CV Mean': result['cv_mean'],
                    'CV Std': result['cv_std']
                })
        else:
            summary = []
            for name, result in self.results.items():
                summary.append({
                    'Model': name,
                    'Train Accuracy': result['train_score'],
                    'Test Accuracy': result['test_score'],
                    'CV Mean': result['cv_mean'],
                    'CV Std': result['cv_std'],
                    'AUC': result['auc_score'],
                    'Precision': result['precision'],
                    'Recall': result['recall'],
                    'F1 Score': result['f1_score'],
                    'Average Precision': result['ap_score'],
                    'MCC': result.get('mcc', 'N/A')
                })
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(output_path / 'model_results_summary.csv', index=False)
        print(f"\nSaved results summary to {output_path / 'model_results_summary.csv'}")
        
        # Save run metadata first (before any error-prone operations)
        self.save_run_metadata(output_path)
        
        # Create visualizations
        try:
            self.visualize_results(output_path)
        except Exception as e:
            print(f"Warning: Error creating visualizations: {e}")
        
        # Additional plots for classification
        if not self.is_regression:
            try:
                self.plot_roc_curves(output_path)
            except Exception as e:
                print(f"Warning: Error creating ROC curves: {e}")
        
        # Feature importance analysis
        try:
            self.feature_importance_analysis(output_path)
        except Exception as e:
            print(f"Warning: Error in feature importance analysis: {e}")
        
    def visualize_results(self, output_path):
        """Create visualizations of model results"""
        # Model comparison plot
        plt.figure(figsize=(10, 6))
        models = list(self.results.keys())
        
        if self.is_regression:
            test_scores = [self.results[m]['r2'] for m in models]
            ylabel = 'R² Score'
            title = 'Model Performance Comparison (PCL Score Prediction)'
        else:
            test_scores = [self.results[m]['test_score'] for m in models]
            ylabel = 'Accuracy'
            title = 'Model Performance Comparison (PTSD Prediction)'
        
        x = np.arange(len(models))
        plt.bar(x, test_scores, alpha=0.8)
        plt.xlabel('Models')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(x, models, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path / 'model_comparison.png', dpi=300)
        plt.close()
        
        if self.is_regression:
            # Scatter plots for regression
            n_models = len(self.results)
            n_cols = 3
            n_rows = (n_models + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.ravel()
            
            for idx, (name, result) in enumerate(self.results.items()):
                if idx < len(axes):
                    y_true = self.y_test
                    y_pred = result['y_pred']
                    
                    axes[idx].scatter(y_true, y_pred, alpha=0.5)
                    axes[idx].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
                    axes[idx].set_xlabel('True PCL Score')
                    axes[idx].set_ylabel('Predicted PCL Score')
                    axes[idx].set_title(f'{name}\nR²={result["test_score"]:.3f}, RMSE={result["rmse"]:.2f}')
                    
            # Hide unused subplots
            for idx in range(len(self.results), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(output_path / 'prediction_scatter_plots.png', dpi=300)
            plt.close()
            
            # Error distribution plots
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.ravel()
            
            for idx, (name, result) in enumerate(self.results.items()):
                if idx < len(axes):
                    errors = self.y_test - result['y_pred']
                    axes[idx].hist(errors, bins=30, alpha=0.7, edgecolor='black')
                    axes[idx].axvline(x=0, color='red', linestyle='--')
                    axes[idx].set_xlabel('Prediction Error (True - Predicted)')
                    axes[idx].set_ylabel('Frequency')
                    axes[idx].set_title(f'{name}\nMAE={result["mae"]:.2f}')
                    
            # Hide unused subplots
            for idx in range(len(self.results), len(axes)):
                axes[idx].set_visible(False)
                
            plt.tight_layout()
            plt.savefig(output_path / 'error_distributions.png', dpi=300)
            plt.close()
        else:
            # Confusion matrices for classification
            n_models = len(self.results)
            n_cols = 3
            n_rows = (n_models + n_cols - 1) // n_cols
            
            # Raw confusion matrices
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.ravel()
            
            for idx, (name, result) in enumerate(self.results.items()):
                if idx < len(axes):
                    cm = result['confusion_matrix']
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
                    axes[idx].set_title(f'{name}')
                    axes[idx].set_xlabel('Predicted')
                    axes[idx].set_ylabel('Actual')
            
            # Hide unused subplots
            for idx in range(len(self.results), len(axes)):
                axes[idx].set_visible(False)
                
            plt.tight_layout()
            plt.savefig(output_path / 'confusion_matrices.png', dpi=300)
            plt.close()
            
            # Normalized confusion matrices (percentages)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.ravel()
            
            for idx, (name, result) in enumerate(self.results.items()):
                if idx < len(axes):
                    cm = result['confusion_matrix']
                    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
                    
                    # Create annotation text with both percentage and count
                    annot_text = np.array([[f'{percent:.1f}%\n({count})' 
                                           for percent, count in zip(row_percent, row_count)] 
                                          for row_percent, row_count in zip(cm_percent, cm)])
                    
                    sns.heatmap(cm_percent, annot=annot_text, fmt='', cmap='Blues', 
                               ax=axes[idx], vmin=0, vmax=100)
                    axes[idx].set_title(f'{name} (%)') 
                    axes[idx].set_xlabel('Predicted')
                    axes[idx].set_ylabel('Actual')
            
            # Hide unused subplots
            for idx in range(len(self.results), len(axes)):
                axes[idx].set_visible(False)
                
            plt.tight_layout()
            plt.savefig(output_path / 'confusion_matrices_percentage.png', dpi=300)
            plt.close()
        
    def run_pipeline(self):
        """Run the complete ML pipeline"""
        print("=" * 50)
        print(f"Starting ML Pipeline for {self.target_column} Prediction")
        print("=" * 50)
        
        # Load and preprocess data
        self.load_data()
        self.preprocess_data()
        
        # Create and train models
        self.create_models()
        self.train_models()
        
        # Identify best model
        if self.is_regression:
            best_model = max(self.results.items(), key=lambda x: x[1]['r2'])[0]
            best_score = self.results[best_model]['r2']
            print(f"\nBest performing model: {best_model} (R² = {best_score:.4f})")
        else:
            best_model = max(self.results.items(), key=lambda x: x[1]['auc_score'])[0]
            best_score = self.results[best_model]['auc_score']
            print(f"\nBest performing model: {best_model} (AUC = {best_score:.4f})")
        
        # Save results
        self.save_results()
        
        print("\n" + "=" * 50)
        print("ML Pipeline completed successfully!")
        print("=" * 50)
        
    def save_run_metadata(self, output_path):
        """Save detailed metadata about this run"""
        metadata = {
            'run_number': self.run_number,
            'run_timestamp': datetime.now().isoformat(),
            'target_variable': self.target_column,
            'task_type': 'regression' if self.is_regression else 'classification',
            'data_info': {
                'source_file': str(self.data_path),
                'total_samples': len(self.df),
                'total_features': len(self.feature_columns),
                'feature_columns': len(self.feature_columns),
                'excluded_columns': ['session_id', 'source_file', 'pcl_score', 'ptsd', 'ptsd_bin'],
                's3_bucket': 'senseye-data-quality',
                's3_path': 'messy_prototyping_saturn_uploads/'
            },
            'target_info': {
                'name': self.target_column,
                'type': 'regression' if self.is_regression else 'classification'
            },
            'ptsd_distribution': {},
            'feature_types': {},
            'model_results': {},
            'training_info': {
                'test_size': self.test_size,
                'random_state': self.random_state,
                'train_samples': len(self.X_train),
                'test_samples': len(self.X_test)
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
            }
        }
        
        # Add target statistics
        if self.is_regression:
            metadata['target_info']['statistics'] = {
                'mean': float(self.y.mean()),
                'std': float(self.y.std()),
                'min': float(self.y.min()),
                'max': float(self.y.max())
            }
        else:
            metadata['target_info']['class_distribution'] = self.y.value_counts().to_dict()
        
        # Add PTSD distribution if available
        if 'ptsd_bin' in self.df.columns:
            ptsd_counts = self.df['ptsd_bin'].value_counts(dropna=False)
            metadata['ptsd_distribution'] = {
                'positive': int(ptsd_counts.get(1, 0)),
                'negative': int(ptsd_counts.get(0, 0)),
                'missing': int(ptsd_counts.get(np.nan, 0))
            }
        
        # Add feature type summary
        for col in self.feature_columns:
            if 'trial' in col and col.startswith('trial'):
                feature_type = col.split('_', 1)[1] if '_' in col else 'unknown'
                if feature_type not in metadata['feature_types']:
                    metadata['feature_types'][feature_type] = 0
                metadata['feature_types'][feature_type] += 1
        
        # Add model results summary
        for name, result in self.results.items():
            if self.is_regression:
                metadata['model_results'][name] = {
                    'train_r2': float(result['train_score']),
                    'test_r2': float(result['test_score']),
                    'rmse': float(result['rmse']),
                    'mae': float(result['mae']),
                    'cv_mean': float(result['cv_mean']),
                    'cv_std': float(result['cv_std'])
                }
            else:
                metadata['model_results'][name] = {
                    'train_accuracy': float(result['train_score']),
                    'test_accuracy': float(result['test_score']),
                    'auc': float(result['auc_score']),
                    'cv_mean': float(result['cv_mean']),
                    'cv_std': float(result['cv_std'])
                }
        
        # Save metadata to JSON
        with open(output_path / 'run_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved run metadata to {output_path / 'run_metadata.json'}")
        
        # Generate HTML report
        try:
            from report_generator import generate_report_for_run
            generate_report_for_run(output_path)
        except Exception as e:
            print(f"Warning: Could not generate HTML report: {e}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ML pipeline for PCL/PTSD prediction')
    parser.add_argument('--data-path', default='processed_sessions/all_sessions_with_pcl.parquet',
                        help='Path to processed data file with PCL scores')
    parser.add_argument('--target', default='ptsd_bin', choices=['ptsd_bin', 'pcl_score'],
                        help='Target column for prediction (ptsd_bin for classification, pcl_score for regression)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size (0-1)')
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
    pipeline = PCLPredictionPipeline(
        data_path=args.data_path,
        target_column=args.target,
        test_size=args.test_size,
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
        print("Running automated pipeline with RFECV and hyperparameter tuning...")
        pipeline.load_data()
        pipeline.preprocess_data()
        pipeline.create_models()
        
        # RFECV
        pipeline.feature_selection_rfecv()
        
        # Train with selected features
        original_X_train = pipeline.X_train_processed
        original_X_test = pipeline.X_test_processed
        pipeline.X_train_processed = pipeline.X_train_processed_rfecv
        pipeline.X_test_processed = pipeline.X_test_processed_rfecv
        
        pipeline.train_models()
        pipeline.evaluate_models()
        
        # Hyperparameter tuning
        pipeline.hyperparameter_tuning(['XGBoost', 'Random Forest', 'Gradient Boosting'])
        
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