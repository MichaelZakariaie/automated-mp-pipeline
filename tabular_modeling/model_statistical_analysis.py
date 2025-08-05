#!/usr/bin/env python3
"""
Statistical analysis of ML model performance
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from pathlib import Path

def load_model_and_data():
    """Load the saved models and test data"""
    # Load the data
    df = pd.read_parquet('processed_sessions/all_sessions_with_pcl.parquet')
    
    # Filter to only rows with PTSD labels
    df = df.dropna(subset=['ptsd_bin'])
    
    # Get feature columns (excluding identifiers and targets)
    exclude_columns = ['session_id', 'source_file', 'pcl_score', 'ptsd', 'ptsd_bin', 'session_id_full']
    feature_columns = []
    
    for col in df.columns:
        if col not in exclude_columns:
            # Only include numeric columns
            if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                feature_columns.append(col)
            elif df[col].dtype == 'object':
                # Convert good/bad to numeric
                if set(df[col].dropna().unique()).issubset({'good', 'bad', ''}):
                    df[col] = df[col].map({'good': 1, 'bad': 0}).fillna(df[col])
                    if df[col].dtype in ['float64', 'int64']:
                        feature_columns.append(col)
    
    X = df[feature_columns]
    y = df['ptsd_bin']
    
    # Load preprocessor
    preprocessor = joblib.load('ml_results/preprocessor_ptsd_bin.pkl')
    
    # We need to recreate the train/test split with the same random state
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Transform test data
    X_test_processed = preprocessor.transform(X_test)
    
    return X_test_processed, y_test, len(y_train), len(y_test)

def calculate_chance_accuracy(y_test):
    """Calculate chance accuracy for baseline comparison"""
    # Majority class baseline
    majority_class = y_test.mode()[0]
    chance_predictions = np.full(len(y_test), majority_class)
    chance_accuracy = accuracy_score(y_test, chance_predictions)
    
    # Random baseline (respecting class distribution)
    class_probs = y_test.value_counts(normalize=True).sort_index().values
    random_predictions = np.random.choice([0, 1], size=len(y_test), p=class_probs)
    random_accuracy = accuracy_score(y_test, random_predictions)
    
    return chance_accuracy, random_accuracy

def binomial_test_accuracy(accuracy, n_samples, chance_level=0.5):
    """
    Perform binomial test to determine if accuracy is significantly better than chance
    """
    n_correct = int(accuracy * n_samples)
    p_value = stats.binom_test(n_correct, n_samples, chance_level, alternative='greater')
    return p_value

def permutation_test(y_true, y_pred, n_permutations=1000):
    """
    Perform permutation test to assess statistical significance
    """
    true_accuracy = accuracy_score(y_true, y_pred)
    
    permuted_accuracies = []
    for _ in range(n_permutations):
        y_permuted = np.random.permutation(y_true)
        perm_accuracy = accuracy_score(y_permuted, y_pred)
        permuted_accuracies.append(perm_accuracy)
    
    # Calculate p-value
    p_value = np.mean(permuted_accuracies >= true_accuracy)
    return p_value, permuted_accuracies

def analyze_models():
    """Analyze all models and compute statistical significance"""
    
    # Load test data
    X_test, y_test, n_train, n_test = load_model_and_data()
    
    # Calculate baselines
    majority_baseline, random_baseline = calculate_chance_accuracy(y_test)
    
    # Model names
    model_names = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'SVM', 'Neural Network']
    model_files = ['random_forest', 'gradient_boosting', 'xgboost', 'svm', 'neural_network']
    
    results = []
    
    print("Analyzing models...")
    
    for name, file in zip(model_names, model_files):
        print(f"\nAnalyzing {name}...")
        
        # Load model
        model = joblib.load(f'ml_results/{file}_model_ptsd_bin.pkl')
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Statistical tests
        # 1. Binomial test against majority baseline
        p_value_binomial_majority = binomial_test_accuracy(accuracy, n_test, majority_baseline)
        
        # 2. Binomial test against random chance (0.5)
        p_value_binomial_chance = binomial_test_accuracy(accuracy, n_test, 0.5)
        
        # 3. Permutation test
        p_value_permutation, _ = permutation_test(y_test, y_pred)
        
        # 4. McNemar's test against majority classifier
        majority_pred = np.ones(len(y_test))  # Predict all as positive (majority class)
        
        # Create contingency table for McNemar's test
        correct_model_wrong_majority = np.sum((y_pred == y_test) & (majority_pred != y_test))
        wrong_model_correct_majority = np.sum((y_pred != y_test) & (majority_pred == y_test))
        
        # McNemar's test
        if correct_model_wrong_majority + wrong_model_correct_majority > 0:
            mcnemar_stat = (abs(correct_model_wrong_majority - wrong_model_correct_majority) - 1)**2 / (correct_model_wrong_majority + wrong_model_correct_majority)
            p_value_mcnemar = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
        else:
            p_value_mcnemar = 1.0
        
        # DeLong test for AUC (simplified version)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'AUC': auc,
            'Majority_Baseline': majority_baseline,
            'Random_Baseline': random_baseline,
            'P_Value_Binomial_vs_Majority': p_value_binomial_majority,
            'P_Value_Binomial_vs_Chance': p_value_binomial_chance,
            'P_Value_Permutation': p_value_permutation,
            'P_Value_McNemar': p_value_mcnemar,
            'N_Train': n_train,
            'N_Test': n_test
        })
    
    return pd.DataFrame(results), majority_baseline, random_baseline

def write_report(results_df, majority_baseline, random_baseline):
    """Write comprehensive statistical report"""
    
    report = []
    report.append("=" * 80)
    report.append("STATISTICAL ANALYSIS OF ML MODELS FOR PTSD PREDICTION")
    report.append("=" * 80)
    report.append("")
    
    # Dataset information
    report.append("DATASET INFORMATION:")
    report.append(f"Total samples: {results_df.iloc[0]['N_Train'] + results_df.iloc[0]['N_Test']}")
    report.append(f"Training samples: {results_df.iloc[0]['N_Train']}")
    report.append(f"Test samples: {results_df.iloc[0]['N_Test']}")
    report.append("")
    
    # Baseline information
    report.append("BASELINE PERFORMANCE:")
    report.append(f"Majority class baseline (always predict positive): {majority_baseline:.4f}")
    report.append(f"Random baseline (respecting class distribution): {random_baseline:.4f}")
    report.append(f"Pure chance (50/50): 0.5000")
    report.append("")
    
    # Statistical significance thresholds
    report.append("STATISTICAL SIGNIFICANCE:")
    report.append("P-value thresholds:")
    report.append("  - p < 0.05  : * (statistically significant)")
    report.append("  - p < 0.01  : ** (highly significant)")
    report.append("  - p < 0.001 : *** (very highly significant)")
    report.append("")
    
    # Model-by-model analysis
    report.append("MODEL PERFORMANCE ANALYSIS:")
    report.append("-" * 80)
    
    for idx, row in results_df.iterrows():
        report.append(f"\n{row['Model'].upper()}:")
        report.append(f"  Accuracy:  {row['Accuracy']:.4f}")
        report.append(f"  Precision: {row['Precision']:.4f}")
        report.append(f"  Recall:    {row['Recall']:.4f}")
        report.append(f"  F1 Score:  {row['F1_Score']:.4f}")
        report.append(f"  AUC:       {row['AUC']:.4f}")
        report.append("")
        
        # Statistical tests
        report.append("  Statistical Tests:")
        
        # Binomial test vs majority
        p_maj = row['P_Value_Binomial_vs_Majority']
        sig_maj = "***" if p_maj < 0.001 else "**" if p_maj < 0.01 else "*" if p_maj < 0.05 else "ns"
        report.append(f"    vs. Majority Baseline ({majority_baseline:.4f}):")
        report.append(f"      Binomial test p-value: {p_maj:.4f} ({sig_maj})")
        
        # Binomial test vs chance
        p_chance = row['P_Value_Binomial_vs_Chance']
        sig_chance = "***" if p_chance < 0.001 else "**" if p_chance < 0.01 else "*" if p_chance < 0.05 else "ns"
        report.append(f"    vs. Pure Chance (0.5000):")
        report.append(f"      Binomial test p-value: {p_chance:.4f} ({sig_chance})")
        
        # Permutation test
        p_perm = row['P_Value_Permutation']
        sig_perm = "***" if p_perm < 0.001 else "**" if p_perm < 0.01 else "*" if p_perm < 0.05 else "ns"
        report.append(f"    Permutation test p-value: {p_perm:.4f} ({sig_perm})")
        
        # McNemar test
        p_mcn = row['P_Value_McNemar']
        sig_mcn = "***" if p_mcn < 0.001 else "**" if p_mcn < 0.01 else "*" if p_mcn < 0.05 else "ns"
        report.append(f"    McNemar test p-value: {p_mcn:.4f} ({sig_mcn})")
        
        # Overall assessment
        report.append("")
        if row['Accuracy'] > majority_baseline and p_maj < 0.05:
            report.append(f"  ✓ Model performs significantly better than majority baseline")
        else:
            report.append(f"  ✗ Model does NOT perform significantly better than majority baseline")
            
        if row['AUC'] > 0.6:
            report.append(f"  ✓ AUC > 0.6 suggests some discriminative ability")
        else:
            report.append(f"  ✗ AUC ≤ 0.6 suggests poor discriminative ability")
    
    report.append("\n" + "=" * 80)
    report.append("SUMMARY:")
    report.append("=" * 80)
    
    # Count significant models
    sig_models = results_df[results_df['P_Value_Binomial_vs_Majority'] < 0.05]
    report.append(f"\nModels significantly better than majority baseline: {len(sig_models)}/{len(results_df)}")
    
    if len(sig_models) > 0:
        report.append("Significant models:")
        for _, row in sig_models.iterrows():
            report.append(f"  - {row['Model']}: Accuracy = {row['Accuracy']:.4f}, p = {row['P_Value_Binomial_vs_Majority']:.4f}")
    
    # Best model
    best_model = results_df.loc[results_df['AUC'].idxmax()]
    report.append(f"\nBest model by AUC: {best_model['Model']} (AUC = {best_model['AUC']:.4f})")
    
    # Overall conclusion
    report.append("\nOVERALL CONCLUSION:")
    if len(sig_models) == 0:
        report.append("⚠️  No models perform significantly better than the majority baseline.")
        report.append("   The features may not be strongly predictive of PTSD status.")
    elif best_model['AUC'] < 0.7:
        report.append("⚠️  While some models show statistical significance, the effect sizes are small.")
        report.append("   AUC values < 0.7 indicate limited clinical utility.")
    else:
        report.append("✓  Some models show both statistical significance and reasonable effect sizes.")
    
    # Write to file
    with open('model_statistical_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    # Also save detailed results as CSV
    results_df.to_csv('model_statistical_results.csv', index=False)
    
    print("\nReport saved to: model_statistical_report.txt")
    print("Detailed results saved to: model_statistical_results.csv")

def main():
    """Run the statistical analysis"""
    results_df, majority_baseline, random_baseline = analyze_models()
    write_report(results_df, majority_baseline, random_baseline)

if __name__ == "__main__":
    main()