#!/usr/bin/env python3
"""
ML Pipeline for analyzing session data with various models
Enhanced with neural networks for overfitting reduction and PCA preprocessing
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Skipping PyTorch-based neural networks.")
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


if TORCH_AVAILABLE:
    class SharedLinearLayer(nn.Module):
        """Linear layer with parameter sharing across multiple inputs"""
        def __init__(self, in_features, out_features, share_groups=4):
            super().__init__()
            self.share_groups = share_groups
            self.group_size = in_features // share_groups
            self.shared_weights = nn.Parameter(torch.randn(self.group_size, out_features))
            self.bias = nn.Parameter(torch.zeros(out_features))
            
        def forward(self, x):
            batch_size = x.size(0)
            x_reshaped = x.view(batch_size, self.share_groups, self.group_size)
            out = torch.matmul(x_reshaped, self.shared_weights)
            out = out.sum(dim=1) + self.bias
            return out


    class LowRankNN(nn.Module):
        """Neural network with low-rank factorization for parameter reduction"""
        def __init__(self, input_dim, hidden_dim=50, rank=10, output_dim=2):
            super().__init__()
            self.fc1_u = nn.Linear(input_dim, rank)
            self.fc1_v = nn.Linear(rank, hidden_dim)
            self.dropout1 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.fc1_u(x)
            x = self.fc1_v(x)
            x = self.relu(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            return x


    class BottleneckNN(nn.Module):
        """Neural network with bottleneck architecture to reduce overfitting"""
        def __init__(self, input_dim, bottleneck_dim=20, hidden_dim=40, output_dim=2):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(hidden_dim, bottleneck_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, output_dim)
            )
            
        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x


    class SharedEmbeddingNN(nn.Module):
        """Neural network with shared embeddings for groups of features"""
        def __init__(self, input_dim, embedding_dim=32, num_groups=8, hidden_dim=64, output_dim=2):
            super().__init__()
            self.num_groups = num_groups
            self.group_size = input_dim // num_groups
            
            # Shared embedding layer
            self.shared_embedding = nn.Linear(self.group_size, embedding_dim)
            self.group_weights = nn.Parameter(torch.randn(num_groups, embedding_dim))
            
            self.fc_layers = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(hidden_dim // 2, output_dim)
            )
            
        def forward(self, x):
            batch_size = x.size(0)
            x_reshaped = x.view(batch_size, self.num_groups, self.group_size)
            
            # Apply shared embedding to each group
            embeddings = []
            for i in range(self.num_groups):
                group_emb = self.shared_embedding(x_reshaped[:, i, :])
                weighted_emb = group_emb * self.group_weights[i]
                embeddings.append(weighted_emb)
            
            # Aggregate embeddings
            x = torch.stack(embeddings, dim=1).mean(dim=1)
            x = self.fc_layers(x)
            return x


    class PyTorchNNClassifier:
        """Wrapper to make PyTorch models compatible with sklearn"""
        def __init__(self, model_class, input_dim, lr=0.001, epochs=100, batch_size=32, **model_kwargs):
            self.model_class = model_class
            self.input_dim = input_dim
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.model_kwargs = model_kwargs
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        def fit(self, X, y):
            self.model = self.model_class(self.input_dim, **self.model_kwargs).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
            self.criterion = nn.CrossEntropyLoss()
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.LongTensor(y).to(self.device)
            
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            self.model.train()
            for epoch in range(self.epochs):
                total_loss = 0
                for batch_X, batch_y in dataloader:
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                    
            return self
            
        def predict(self, X):
            self.model.eval()
            X_tensor = torch.FloatTensor(X).to(self.device)
            with torch.no_grad():
                outputs = self.model(X_tensor)
                _, predicted = torch.max(outputs, 1)
            return predicted.cpu().numpy()
            
        def predict_proba(self, X):
            self.model.eval()
            X_tensor = torch.FloatTensor(X).to(self.device)
            with torch.no_grad():
                outputs = self.model(X_tensor)
                probs = torch.softmax(outputs, dim=1)
            return probs.cpu().numpy()
            
        def score(self, X, y):
            predictions = self.predict(X)
            return (predictions == y).mean()


class SessionMLPipeline:
    """ML Pipeline for session data analysis"""
    
    def __init__(self, data_path, target_column='pcl_score', test_size=0.2, random_state=42, 
                 include_models=None, exclude_models=None, sample_fraction=1.0):
        """
        Initialize the ML pipeline
        
        Args:
            data_path: Path to the processed data file (parquet or csv)
            target_column: Column to use as target variable
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.include_models = include_models
        self.exclude_models = exclude_models if exclude_models else []
        self.sample_fraction = sample_fraction
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load and prepare the data"""
        print(f"Loading data from {self.data_path}...")
        
        if self.data_path.suffix == '.parquet':
            self.df = pd.read_parquet(self.data_path)
        else:
            self.df = pd.read_csv(self.data_path)
            
        print(f"Data shape: {self.df.shape}")
        
        # Remove rows with missing target values
        if self.df[self.target_column].isna().any():
            print(f"Removing {self.df[self.target_column].isna().sum()} rows with missing {self.target_column}")
            self.df = self.df.dropna(subset=[self.target_column])
            print(f"Data shape after removing missing targets: {self.df.shape}")
        
        # Remove identifier columns
        id_columns = ['session_id', 'source_file']
        self.feature_columns = [col for col in self.df.columns if col not in id_columns + [self.target_column]]
        
        # Prepare features and target
        self.X = self.df[self.feature_columns]
        self.y = self.df[self.target_column]
        
        # Handle categorical features
        print("\nEncoding categorical features...")
        categorical_cols = []
        for col in self.X.columns:
            if self.X[col].dtype == 'object':
                categorical_cols.append(col)
                # Get unique values
                unique_vals = self.X[col].dropna().unique()
                # Map good/bad to 1/0, handle empty strings and None
                if set(unique_vals).issubset({'good', 'bad', ''}):
                    self.X[col] = self.X[col].map({'good': 1, 'bad': 0, '': None, None: None})
                else:
                    # For other categorical variables, use pandas get_dummies or drop
                    print(f"Warning: Column {col} has values: {unique_vals}")
                    # Convert to numeric if possible, otherwise drop
                    try:
                        self.X[col] = pd.to_numeric(self.X[col], errors='coerce')
                    except:
                        print(f"Dropping column {col} - unable to convert to numeric")
                        self.X = self.X.drop(columns=[col])
                        self.feature_columns.remove(col)
        
        if categorical_cols:
            print(f"Processed {len(categorical_cols)} categorical columns")
        
        # Handle categorical target if needed
        if self.y.dtype == 'object':
            self.y = self.y.map({'good': 1, 'bad': 0})
        
        print(f"Features: {len(self.feature_columns)} columns")
        print(f"Target distribution:\n{self.y.value_counts()}")
        
    def preprocess_data(self):
        """Preprocess the data"""
        print("\nPreprocessing data...")
        
        # Create preprocessing pipeline
        self.preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Split data
        # Check if target is continuous (regression) or categorical (classification)
        n_unique = self.y.nunique()
        is_regression = n_unique > 10 or self.y.dtype in ['float64', 'float32']
        
        if is_regression:
            # For regression, don't use stratify
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=self.test_size, random_state=self.random_state
            )
            self.task_type = 'regression'
        else:
            # For classification, use stratify
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=self.test_size, random_state=self.random_state, stratify=self.y
            )
            self.task_type = 'classification'
        
        # Fit and transform
        self.X_train_processed = self.preprocessor.fit_transform(self.X_train)
        self.X_test_processed = self.preprocessor.transform(self.X_test)
        
        print(f"Training set: {self.X_train_processed.shape}")
        print(f"Test set: {self.X_test_processed.shape}")
        
    def create_models(self):
        """Create various ML models"""
        print("\nCreating models...")
        print(f"Task type: {self.task_type}")
        
        # Get input dimension for PyTorch models
        input_dim = self.X_train_processed.shape[1]
        
        if self.task_type == 'regression':
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
                    n_jobs=-1
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
                
                'SVM': SVC(
                    kernel='rbf',
                    gamma='scale',  # Better for non-linear data
                    C=1.0,
                    probability=True,
                    random_state=self.random_state
                ),
                
                'SVM_Poly': SVC(
                    kernel='poly',
                    degree=3,
                    gamma='scale',
                    C=1.0,
                    probability=True,
                    random_state=self.random_state
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
                    alpha=0.001,  # L2 regularization
                    learning_rate='adaptive',
                    max_iter=1000,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=self.random_state
                ),
                
                'Wide NN': MLPClassifier(
                    hidden_layer_sizes=(300,),
                    activation='tanh',  # Different activation
                    solver='lbfgs',  # Better for small datasets
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
        
        # Add PyTorch models if available
        if TORCH_AVAILABLE:
            self.models.update({
                'Low Rank NN': PyTorchNNClassifier(
                    LowRankNN, 
                    input_dim=input_dim,
                    hidden_dim=30,
                    rank=8,
                    lr=0.001,
                    epochs=50
                ),
                
                'Bottleneck NN': PyTorchNNClassifier(
                    BottleneckNN,
                    input_dim=input_dim,
                    bottleneck_dim=15,
                    hidden_dim=30,
                    lr=0.001,
                    epochs=50
                ),
                
                'Shared Embedding NN': PyTorchNNClassifier(
                    SharedEmbeddingNN,
                    input_dim=input_dim,
                    embedding_dim=24,
                    num_groups=8,
                    hidden_dim=48,
                    lr=0.001,
                    epochs=50
                )
            })
        
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
        
        # Create PCA versions of tree-based models
        self.create_pca_models()
    
    def create_pca_models(self):
        """Create PCA-preprocessed versions of tree-based models"""
        print("\nCreating PCA models with 30 dimensions...")
        
        # Create PCA transformer
        self.pca = PCA(n_components=30, random_state=self.random_state)
        
        # Fit PCA on training data
        self.X_train_pca = self.pca.fit_transform(self.X_train_processed)
        self.X_test_pca = self.pca.transform(self.X_test_processed)
        
        print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        # Add PCA versions of models
        if self.task_type == 'regression':
            self.pca_models = {
                'Random Forest + PCA': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                
                'Gradient Boosting + PCA': GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=self.random_state
                ),
                
                'XGBoost + PCA': xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=self.random_state,
                    eval_metric='rmse'
                )
            }
        else:
            self.pca_models = {
                'Random Forest + PCA': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                
                'Gradient Boosting + PCA': GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=self.random_state
                ),
                
                'XGBoost + PCA': xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=self.random_state,
                    eval_metric='logloss'
                )
            }
        
        # Add PCA models to main models dict (only if the base model is included)
        if self.include_models:
            filtered_pca_models = {name: model for name, model in self.pca_models.items()
                                 if any(base in name for base in self.include_models)}
            self.models.update(filtered_pca_models)
        elif self.exclude_models:
            filtered_pca_models = {name: model for name, model in self.pca_models.items()
                                 if not any(base in name for base in self.exclude_models)}
            self.models.update(filtered_pca_models)
        else:
            self.models.update(self.pca_models)
        
    def train_models(self):
        """Train all models"""
        print("\nTraining models...")
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Use PCA data for PCA models
            if 'PCA' in name:
                X_train = self.X_train_pca
                X_test = self.X_test_pca
            else:
                X_train = self.X_train_processed
                X_test = self.X_test_processed
            
            # Train model
            model.fit(X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            if self.task_type == 'regression':
                # Calculate regression metrics
                train_score = model.score(X_train, self.y_train)  # R2 score
                test_score = model.score(X_test, self.y_test)  # R2 score
                mse = mean_squared_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(self.y_test, y_pred)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, self.y_train, cv=5, scoring='r2')
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'train_score': train_score,
                    'test_score': test_score,
                    'cv_scores': cv_scores,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_pred': y_pred,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae
                }
                
                print(f"  Train R²: {train_score:.4f}")
                print(f"  Test R²: {test_score:.4f}")
                print(f"  RMSE: {rmse:.4f}")
                print(f"  MAE: {mae:.4f}")
                print(f"  CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            else:
                # Classification metrics
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                else:
                    y_pred_proba = None
                
                train_score = model.score(X_train, self.y_train)
                test_score = model.score(X_test, self.y_test)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, self.y_train, cv=5)
                
                # Store results
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
                    'mcc': matthews_corrcoef(self.y_test, y_pred)
                }
                
                # Calculate AUC if binary classification
                if len(np.unique(self.y)) == 2 and y_pred_proba is not None:
                    self.results[name]['auc_score'] = roc_auc_score(self.y_test, y_pred_proba)
                
                print(f"  Train accuracy: {train_score:.4f}")
                print(f"  Test accuracy: {test_score:.4f}")
                print(f"  CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
    def hyperparameter_tuning(self, model_names=None, use_random_search=True, n_iter=50):
        """Perform hyperparameter tuning for specified models"""
        if model_names is None:
            # Default to tuning the best performing models
            model_names = ['XGBoost', 'Random Forest', 'Gradient Boosting', 'SVM']
        
        param_distributions = {
            'XGBoost': {
                'n_estimators': [50, 100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 10],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.5, 0.7, 0.9, 1.0],
                'gamma': [0, 0.1, 0.5, 1],
                'reg_alpha': [0, 0.001, 0.01, 0.1],
                'reg_lambda': [1, 1.5, 2, 3]
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.5, 0.7],
                'bootstrap': [True, False]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 10],
                'subsample': [0.6, 0.8, 1.0],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'degree': [2, 3, 4]  # for poly kernel
            },
            'Logistic Regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga'],
                'l1_ratio': [0.1, 0.5, 0.9]  # for elasticnet
            },
            'Neural Network': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100), (100, 50, 25)],
                'activation': ['relu', 'tanh', 'logistic'],
                'solver': ['adam', 'lbfgs', 'sgd'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'batch_size': ['auto', 32, 64, 128]
            }
        }
        
        tuned_models = {}
        scoring = 'r2' if self.task_type == 'regression' else 'f1'
        
        for model_name in model_names:
            if model_name in self.models and model_name in param_distributions:
                print(f"\n{'='*50}")
                print(f"Tuning {model_name}...")
                print(f"{'='*50}")
                
                base_model = self.models[model_name]
                param_dist = param_distributions[model_name]
                
                # Use RandomizedSearchCV for efficiency
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
                
                print(f"\nBest parameters: {search.best_params_}")
                print(f"Best CV score: {search.best_score_:.4f}")
                
                # Store tuned model
                tuned_models[f'{model_name}_Tuned'] = search.best_estimator_
                
                # Evaluate on test set
                test_score = search.score(self.X_test_processed, self.y_test)
                print(f"Test score: {test_score:.4f}")
        
        # Add tuned models to the model list
        self.models.update(tuned_models)
        
        # Re-evaluate all models including tuned ones
        self.train_models()
        
        return tuned_models
    
    def feature_selection_rfecv(self, estimator=None, step=1, min_features=5):
        """Perform Recursive Feature Elimination with Cross-Validation"""
        print("\n" + "="*50)
        print("Running RFECV Feature Selection...")
        print("="*50)
        
        # Use Random Forest as default estimator
        if estimator is None:
            if self.task_type == 'regression':
                estimator = RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)
            else:
                estimator = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        
        # Create RFECV
        rfecv = RFECV(
            estimator=estimator,
            step=step,
            cv=5,
            scoring='r2' if self.task_type == 'regression' else 'f1',
            min_features_to_select=min_features,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit RFECV
        rfecv.fit(self.X_train_processed, self.y_train)
        
        print(f"\nOptimal number of features: {rfecv.n_features_}")
        print(f"Features selected: {np.sum(rfecv.support_)} out of {len(rfecv.support_)}")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.plot(range(min_features, len(rfecv.grid_scores_) + min_features), rfecv.grid_scores_)
        plt.xlabel('Number of features selected')
        plt.ylabel('Cross validation score')
        plt.title('RFECV - Optimal number of features')
        plt.grid(True)
        plt.tight_layout()
        
        # Transform data with selected features
        self.X_train_processed_rfecv = rfecv.transform(self.X_train_processed)
        self.X_test_processed_rfecv = rfecv.transform(self.X_test_processed)
        
        # Store RFECV transformer
        self.rfecv = rfecv
        
        # Get selected feature names
        selected_features = np.array(self.feature_columns)[rfecv.support_]
        print(f"\nSelected features: {list(selected_features[:10])}...")  # Show first 10
        
        return rfecv
    
    def run_automated_ml_pipeline(self, use_rfecv=True, use_tuning=True, top_models=3):
        """Run a complete automated ML pipeline with feature selection and hyperparameter tuning"""
        print("\n" + "="*60)
        print("RUNNING AUTOMATED ML PIPELINE")
        print("="*60)
        
        # Step 1: Feature Selection with RFECV
        if use_rfecv:
            print("\nStep 1: Feature Selection with RFECV")
            self.feature_selection_rfecv()
            
            # Create models with selected features
            print("\nTraining models with selected features...")
            original_X_train = self.X_train_processed
            original_X_test = self.X_test_processed
            
            self.X_train_processed = self.X_train_processed_rfecv
            self.X_test_processed = self.X_test_processed_rfecv
            
            # Train models with selected features
            self.train_models()
            
            # Restore original data for comparison
            self.X_train_processed = original_X_train
            self.X_test_processed = original_X_test
        else:
            # Train models without feature selection
            self.train_models()
        
        # Step 2: Select top performing models
        print(f"\nStep 2: Selecting top {top_models} models for hyperparameter tuning")
        model_scores = [(name, result['test_score']) for name, result in self.results.items()]
        model_scores.sort(key=lambda x: x[1], reverse=True)
        top_model_names = [name for name, _ in model_scores[:top_models]]
        print(f"Top models: {top_model_names}")
        
        # Step 3: Hyperparameter Tuning
        if use_tuning:
            print("\nStep 3: Hyperparameter Tuning")
            self.hyperparameter_tuning(model_names=top_model_names, use_random_search=True, n_iter=30)
        
        # Step 4: Final evaluation and report
        print("\nStep 4: Final Model Evaluation")
        print("="*60)
        
        # Sort models by performance
        final_scores = [(name, result['test_score']) for name, result in self.results.items()]
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        print("\nFinal Model Rankings:")
        for i, (name, score) in enumerate(final_scores[:10], 1):
            print(f"{i}. {name}: {score:.4f}")
        
        return self.results
            
    def feature_importance_analysis(self):
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
        plt.title('Top 20 Most Important Features (Random Forest)')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300)
        plt.close()
        
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))
        
        return feature_importance
        
    def save_results(self, output_dir='ml_results'):
        """Save models and results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
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
        summary = []
        for name, result in self.results.items():
            if self.task_type == 'regression':
                summary.append({
                    'Model': name,
                    'Train R²': result['train_score'],
                    'Test R²': result['test_score'],
                    'RMSE': result['rmse'],
                    'MAE': result['mae'],
                    'CV Mean R²': result['cv_mean'],
                    'CV Std': result['cv_std']
                })
            else:
                summary.append({
                    'Model': name,
                    'Train Accuracy': result['train_score'],
                    'Test Accuracy': result['test_score'],
                    'CV Mean': result['cv_mean'],
                    'CV Std': result['cv_std'],
                    'AUC': result.get('auc_score', 'N/A'),
                    'MCC': result.get('mcc', 'N/A')
                })
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(output_path / 'model_results_summary.csv', index=False)
        print(f"\nSaved results summary to {output_path / 'model_results_summary.csv'}")
        
        # Create visualization of results
        self.visualize_results(output_path)
        
        return output_path
        
    def visualize_results(self, output_path):
        """Create visualizations of model results"""
        # Model comparison plot
        plt.figure(figsize=(10, 6))
        models = list(self.results.keys())
        train_scores = [self.results[m]['train_score'] for m in models]
        test_scores = [self.results[m]['test_score'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, train_scores, width, label='Train', alpha=0.8)
        plt.bar(x + width/2, test_scores, width, label='Test', alpha=0.8)
        
        plt.xlabel('Models')
        plt.ylabel('R² Score' if self.task_type == 'regression' else 'Accuracy')
        plt.title('Model Performance Comparison')
        plt.xticks(x, models, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / 'model_comparison.png', dpi=300)
        plt.close()
        
        if self.task_type == 'regression':
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
        
    def run_pipeline(self):
        """Run the complete ML pipeline"""
        print("=" * 50)
        print("Starting ML Pipeline")
        print("=" * 50)
        
        # Load and preprocess data
        self.load_data()
        self.preprocess_data()
        
        # Create and train models
        self.create_models()
        self.train_models()
        
        # Feature importance
        self.feature_importance_analysis()
        
        # Hyperparameter tuning for best model
        best_model = max(self.results.items(), key=lambda x: x[1]['test_score'])[0]
        print(f"\nBest performing model: {best_model} (Test accuracy: {self.results[best_model]['test_score']:.4f})")
        
        # Save results
        output_path = self.save_results()
        
        # Generate unified HTML report
        from report_generator import generate_report_for_run
        report_path = generate_report_for_run(output_path, use_unified=True)
        
        print("\n" + "=" * 50)
        print("ML Pipeline completed successfully!")
        print(f"Report saved: {report_path}")
        print("=" * 50)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ML pipeline on processed session data')
    parser.add_argument('--data-path', default='processed_sessions/all_sessions_with_pcl.parquet',
                        help='Path to processed data file')
    parser.add_argument('--target', default='pcl_score',
                        help='Target column for prediction')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size (0-1)')
    parser.add_argument('--tune', action='store_true',
                        help='Perform hyperparameter tuning')
    parser.add_argument('--auto', action='store_true',
                        help='Run automated ML pipeline with RFECV and tuning')
    parser.add_argument('--rfecv', action='store_true',
                        help='Perform RFECV feature selection')
    parser.add_argument('--sample-fraction', type=float, default=1.0,
                        help='Fraction of data to use')
    parser.add_argument('--include-models', nargs='+', default=None,
                        help='List of models to include')
    parser.add_argument('--exclude-models', nargs='+', default=None,
                        help='List of models to exclude')
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = SessionMLPipeline(
        data_path=args.data_path,
        target_column=args.target,
        test_size=args.test_size,
        sample_fraction=args.sample_fraction,
        include_models=args.include_models,
        exclude_models=args.exclude_models
    )
    
    if args.auto:
        # Run automated pipeline
        pipeline.run_automated_ml_pipeline(use_rfecv=True, use_tuning=True, top_models=3)
    else:
        # Run standard pipeline
        pipeline.run_pipeline()
        
        # Optional RFECV
        if args.rfecv:
            pipeline.feature_selection_rfecv()
        
        # Optional hyperparameter tuning
        if args.tune:
            pipeline.hyperparameter_tuning(['XGBoost', 'Random Forest', 'Gradient Boosting'])

if __name__ == "__main__":
    main()