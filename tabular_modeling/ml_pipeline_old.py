#!/usr/bin/env python3
"""
ML Pipeline for analyzing session data with various models
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
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
    
    def __init__(self, data_path, target_column='session_saccade_data_quality_pog', test_size=0.2, random_state=42):
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
        
        # Remove identifier columns
        id_columns = ['session_id', 'source_file']
        self.feature_columns = [col for col in self.df.columns if col not in id_columns + [self.target_column]]
        
        # Prepare features and target
        self.X = self.df[self.feature_columns]
        self.y = self.df[self.target_column]
        
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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state, stratify=self.y
        )
        
        # Fit and transform
        self.X_train_processed = self.preprocessor.fit_transform(self.X_train)
        self.X_test_processed = self.preprocessor.transform(self.X_test)
        
        print(f"Training set: {self.X_train_processed.shape}")
        print(f"Test set: {self.X_test_processed.shape}")
        
    def create_models(self):
        """Create various ML models"""
        print("\nCreating models...")
        
        # Get input dimension for PyTorch models
        input_dim = self.X_train_processed.shape[1]
        
        self.models = {
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
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            
            'SVM': SVC(
                kernel='rbf',
                probability=True,
                random_state=self.random_state
            ),
            
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                max_iter=500,
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
                use_label_encoder=False,
                eval_metric='logloss'
            )
        }
        
        # Add PCA models to main models dict
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
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
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
                'confusion_matrix': confusion_matrix(self.y_test, y_pred)
            }
            
            # Calculate AUC if binary classification
            if len(np.unique(self.y)) == 2:
                self.results[name]['auc_score'] = roc_auc_score(self.y_test, y_pred_proba)
            
            print(f"  Train accuracy: {train_score:.4f}")
            print(f"  Test accuracy: {test_score:.4f}")
            print(f"  CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
    def hyperparameter_tuning(self, model_name='XGBoost'):
        """Perform hyperparameter tuning for a specific model"""
        print(f"\nPerforming hyperparameter tuning for {model_name}...")
        
        param_grids = {
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
        
        if model_name in param_grids:
            base_model = self.models[model_name]
            param_grid = param_grids[model_name]
            
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(self.X_train_processed, self.y_train)
            
            print(f"\nBest parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            
            # Update model with best parameters
            self.models[f'{model_name}_tuned'] = grid_search.best_estimator_
            
            # Evaluate tuned model
            test_score = grid_search.score(self.X_test_processed, self.y_test)
            print(f"Test score with best parameters: {test_score:.4f}")
            
            return grid_search
            
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
            summary.append({
                'Model': name,
                'Train Accuracy': result['train_score'],
                'Test Accuracy': result['test_score'],
                'CV Mean': result['cv_mean'],
                'CV Std': result['cv_std'],
                'AUC': result.get('auc_score', 'N/A')
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(output_path / 'model_results_summary.csv', index=False)
        print(f"\nSaved results summary to {output_path / 'model_results_summary.csv'}")
        
        # Create visualization of results
        self.visualize_results(output_path)
        
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
        plt.ylabel('Accuracy')
        plt.title('Model Performance Comparison')
        plt.xticks(x, models, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / 'model_comparison.png', dpi=300)
        plt.close()
        
        # Confusion matrices (raw numbers)
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
        
        # Confusion matrices (percentages)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.ravel()
        
        for idx, (name, result) in enumerate(self.results.items()):
            if idx < len(axes):
                cm = result['confusion_matrix']
                # Convert to percentages
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
        self.save_results()
        
        print("\n" + "=" * 50)
        print("ML Pipeline completed successfully!")
        print("=" * 50)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ML pipeline on processed session data')
    parser.add_argument('--data-path', default='processed_sessions/all_sessions_processed.parquet',
                        help='Path to processed data file')
    parser.add_argument('--target', default='session_saccade_data_quality_pog',
                        help='Target column for prediction')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size (0-1)')
    parser.add_argument('--tune', action='store_true',
                        help='Perform hyperparameter tuning')
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = SessionMLPipeline(
        data_path=args.data_path,
        target_column=args.target,
        test_size=args.test_size
    )
    
    pipeline.run_pipeline()
    
    # Optional hyperparameter tuning
    if args.tune:
        pipeline.hyperparameter_tuning('XGBoost')
        pipeline.hyperparameter_tuning('Random Forest')

if __name__ == "__main__":
    main()