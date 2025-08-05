#!/usr/bin/env python3
"""
HTML Report Generator for ML Pipeline Results
Generates comprehensive reports with embedded images and formatted text
"""

import base64
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import os
import sys

# Report server deprecated - files now saved for manual SCP transfer

class ReportGenerator:
    """Generate HTML reports for ML pipeline results"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.html_content = []
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
    def image_to_base64(self, image_path):
        """Convert image file to base64 string for embedding in HTML"""
        if not Path(image_path).exists():
            return None
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    
    def figure_to_base64(self, fig):
        """Convert matplotlib figure to base64 string"""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        buf.close()
        plt.close(fig)
        return img_base64
    
    def add_header(self, title, subtitle=None):
        """Add header section to report"""
        self.html_content.append(f"""
        <div class="header">
            <h1>{title}</h1>
            {f'<h2>{subtitle}</h2>' if subtitle else ''}
            <p class="timestamp">Generated on: {self.timestamp}</p>
        </div>
        """)
    
    def add_section(self, title, level=2):
        """Add section header"""
        self.html_content.append(f"<h{level}>{title}</h{level}>")
    
    def add_text(self, text, style=""):
        """Add text paragraph"""
        self.html_content.append(f'<p style="{style}">{text}</p>')
    
    def add_code_block(self, code):
        """Add code block"""
        self.html_content.append(f'<pre class="code-block">{code}</pre>')
    
    def add_table(self, df, title=None, index=False):
        """Add pandas DataFrame as HTML table"""
        if title:
            self.html_content.append(f'<h3>{title}</h3>')
        
        # Format numeric columns
        if isinstance(df, pd.DataFrame):
            for col in df.select_dtypes(include=[np.number]).columns:
                if 'Mean' in col or 'Std' in col or 'Score' in col or 'R²' in col:
                    df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "")
                elif 'RMSE' in col or 'MAE' in col or 'MSE' in col:
                    df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
        
        table_html = df.to_html(index=index, classes='data-table')
        self.html_content.append(table_html)
    
    def add_image(self, image_path, title=None, width="100%"):
        """Add image from file"""
        img_base64 = self.image_to_base64(image_path)
        if img_base64:
            if title:
                self.html_content.append(f'<h3>{title}</h3>')
            self.html_content.append(f'''
            <div class="image-container">
                <img src="data:image/png;base64,{img_base64}" style="width: {width}; max-width: 100%;">
            </div>
            ''')
    
    def add_figure(self, fig, title=None, width="100%"):
        """Add matplotlib figure directly"""
        img_base64 = self.figure_to_base64(fig)
        if title:
            self.html_content.append(f'<h3>{title}</h3>')
        self.html_content.append(f'''
        <div class="image-container">
            <img src="data:image/png;base64,{img_base64}" style="width: {width}; max-width: 100%;">
        </div>
        ''')
    
    def add_confusion_matrix_table(self, cm, model_name, classes=['Negative', 'Positive']):
        """Add confusion matrix as HTML table with percentages and raw counts"""
        total = cm.sum()
        
        html = f'<div style="margin: 20px 0; background-color: #f8f9fa; padding: 20px; border-radius: 5px;">\n'
        html += f'<h4 style="text-align: center; color: #2c3e50;">{model_name} - Confusion Matrix</h4>\n'
        html += '<table style="margin: auto; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-collapse: collapse;">\n'
        
        # Header row
        html += '<tr>'
        html += '<td style="background-color: #34495e; color: white; padding: 12px; font-weight: bold;"></td>'
        for pred_class in classes:
            html += f'<td style="background-color: #34495e; color: white; padding: 12px; text-align: center; font-weight: bold;">Predicted {pred_class}</td>'
        html += '</tr>\n'
        
        # Data rows
        for i, true_class in enumerate(classes):
            html += f'<tr><td style="background-color: #34495e; color: white; padding: 12px; font-weight: bold;">Actual {true_class}</td>'
            for j in range(len(classes)):
                count = cm[i, j]
                percentage = (count / total) * 100
                cell_color = '#d4edda' if i == j else '#f8d7da'
                html += f'<td style="text-align: center; padding: 12px; background-color: {cell_color}; font-weight: bold;">'
                html += f'{percentage:.1f}% ({count})</td>'
            html += '</tr>\n'
        
        html += '</table>\n'
        
        # Add metrics below the table
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / total
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            html += '<div style="margin-top: 15px; text-align: center;">'
            html += f'<strong>Accuracy:</strong> {accuracy:.3f} | '
            html += f'<strong>Precision:</strong> {precision:.3f} | '
            html += f'<strong>Recall:</strong> {recall:.3f} | '
            html += f'<strong>F1:</strong> {f1:.3f}'
            html += '</div>'
        
        html += '</div>\n'
        
        self.html_content.append(html)
    
    def generate_kfold_report(self, metadata_path, results_summary_path, detailed_results_path=None):
        """Generate report for k-fold cross-validation results"""
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load results
        results_df = pd.read_csv(results_summary_path)
        
        # Add header
        task_type = metadata['task_type']
        target = metadata['target_variable']
        self.add_header(
            f"K-Fold Cross-Validation Report: {target.upper()}",
            f"{task_type.capitalize()} Task"
        )
        
        # Add overview section
        self.add_section("Overview")
        self.add_text(f"""
        This report summarizes the results of k-fold cross-validation for {target} prediction.
        The analysis used {metadata['validation_settings']['n_folds']} folds repeated 
        {metadata['validation_settings']['n_iterations']} times for a total of 
        {metadata['validation_settings']['total_evaluations']} evaluations per model.
        """)
        
        # Add data information
        self.add_section("Dataset Information")
        self.add_text(f"""
        <strong>Source:</strong> {metadata['data_info']['source_file']}<br>
        <strong>Total Samples:</strong> {metadata['data_info']['total_samples']}<br>
        <strong>Total Features:</strong> {metadata['data_info']['total_features']}<br>
        <strong>Target Variable:</strong> {target} ({task_type})
        """)
        
        # Add PCL filtering information if applicable
        if 'pcl_filtering' in metadata and metadata['pcl_filtering']['remove_intermediate']:
            self.add_text(f"""
            <strong>PCL Score Filtering:</strong> Removed intermediate scores<br>
            <strong>Excluded Range:</strong> {metadata['pcl_filtering']['excluded_range']}<br>
            <strong>PTSD Threshold:</strong> {metadata['pcl_filtering']['ptsd_threshold']}<br>
            <strong>Buffer:</strong> ±{metadata['pcl_filtering']['buffer']} points
            """)
        
        # Add results summary table
        self.add_section("Model Performance Summary")
        self.add_table(results_df, "Average Performance Across All Folds")
        
        # Add visualizations
        self.add_section("Visualizations")
        
        # Model comparison
        model_comparison_path = self.output_dir / 'kfold_model_comparison.png'
        if model_comparison_path.exists():
            self.add_image(model_comparison_path, "Model Performance Comparison")
        
        # Score distributions
        score_dist_path = self.output_dir / 'kfold_score_distributions.png'
        if score_dist_path.exists():
            self.add_image(score_dist_path, "Score Distribution Across Folds")
        
        # For classification tasks, add confusion matrices and ROC curves
        if task_type == 'classification':
            # Confusion matrices
            cm_path = self.output_dir / 'kfold_confusion_matrices.png'
            if cm_path.exists():
                self.add_image(cm_path, "Aggregate Confusion Matrices from K-Fold")
            
            # ROC curves
            roc_path = self.output_dir / 'kfold_roc_curves.png'
            if roc_path.exists():
                self.add_image(roc_path, "Aggregate ROC Curves from K-Fold")
            elif detailed_results_path:
                # If no actual ROC curves, generate synthetic ones
                self.add_synthetic_roc_curve(detailed_results_path, metadata)
        
        # Add interpretation
        self.add_section("Interpretation")
        
        if task_type == 'regression':
            best_model = results_df.loc[results_df['R² Mean'].astype(float).idxmax()]
            self.add_text(f"""
            The best performing model was <strong>{best_model['Model']}</strong> with an average R² score of 
            {float(best_model['R² Mean']):.4f} ± {float(best_model['R² Std']):.4f}. 
            However, all models show negative R² scores, indicating they perform worse than a simple mean baseline,
            suggesting significant challenges in predicting PCL scores from the current features.
            """)
        else:
            best_model = results_df.loc[results_df['AUC Mean'].astype(float).idxmax()]
            self.add_text(f"""
            The best performing model was <strong>{best_model['Model']}</strong> with an average AUC of 
            {float(best_model['AUC Mean']):.4f} ± {float(best_model['AUC Std']):.4f}. 
            An AUC near 0.5 indicates performance close to random chance, suggesting the models
            struggle to distinguish between PTSD positive and negative cases using the current features.
            """)
    
    def add_synthetic_roc_curve(self, detailed_results_path, metadata):
        """Generate synthetic average ROC curves from k-fold results"""
        # This is a simplified synthetic ROC curve based on average AUC scores
        # In a real implementation, you'd aggregate the actual predictions
        
        self.add_section("Synthetic ROC Curves", level=3)
        self.add_text("""
        <em>Note: These are synthetic ROC curves generated based on average AUC scores from k-fold validation.
        They represent the expected average performance across folds.</em>
        """)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Get AUC scores from metadata
        model_results = metadata['model_results_summary']
        
        # Generate synthetic ROC curves
        for model_name, results in model_results.items():
            auc_score = results['auc_mean']
            
            # Generate synthetic ROC curve points
            # This creates a smooth curve that would result in the given AUC
            if auc_score >= 0.5:
                # Curve above diagonal
                fpr = np.linspace(0, 1, 100)
                # Use a power function to create a curve with the desired AUC
                power = -np.log(2 * (1 - auc_score)) / np.log(0.5) if auc_score < 1 else 0.1
                tpr = fpr ** (1 / (power + 1))
            else:
                # Curve below diagonal
                fpr = np.linspace(0, 1, 100)
                power = -np.log(2 * auc_score) / np.log(0.5)
                tpr = 1 - (1 - fpr) ** (1 / (power + 1))
            
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f} ± {results["auc_std"]:.3f})')
        
        # Add diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Synthetic Average ROC Curves from K-Fold Validation')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        self.add_figure(fig, width="80%")
    
    def generate_regular_report(self, metadata_path, results_summary_path):
        """Generate report for regular train/test split results"""
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load results
        results_df = pd.read_csv(results_summary_path)
        
        # Add header
        task_type = metadata['task_type']
        target = metadata['target_variable']
        self.add_header(
            f"ML Pipeline Report: {target.upper()}",
            f"{task_type.capitalize()} Task"
        )
        
        # Add overview
        self.add_section("Overview")
        self.add_text(f"""
        This report summarizes the machine learning pipeline results for {target} prediction.
        The analysis used a {int(metadata['training_info']['test_size']*100)}% test set 
        with {metadata['training_info']['train_samples']} training samples and 
        {metadata['training_info']['test_samples']} test samples.
        """)
        
        # Add data information
        self.add_section("Dataset Information")
        
        # Basic info
        self.add_text(f"""
        <strong>Source:</strong> {metadata['data_info']['source_file']}<br>
        <strong>Total Samples:</strong> {metadata['data_info']['total_samples']}<br>
        <strong>Total Features:</strong> {metadata['data_info']['total_features']}<br>
        <strong>S3 Bucket:</strong> {metadata['data_info']['s3_bucket']}<br>
        <strong>S3 Path:</strong> {metadata['data_info']['s3_path']}
        """)
        
        # PTSD distribution
        if 'ptsd_distribution' in metadata:
            self.add_text(f"""
            <strong>PTSD Distribution:</strong><br>
            - Positive: {metadata['ptsd_distribution']['positive']} samples<br>
            - Negative: {metadata['ptsd_distribution']['negative']} samples<br>
            - Missing: {metadata['ptsd_distribution']['missing']} samples
            """)
        
        # PCL filtering information if applicable
        if 'pcl_filtering' in metadata and metadata['pcl_filtering']['remove_intermediate']:
            self.add_text(f"""
            <strong>PCL Score Filtering:</strong> Removed intermediate scores<br>
            <strong>Excluded Range:</strong> {metadata['pcl_filtering']['excluded_range']}<br>
            <strong>PTSD Threshold:</strong> {metadata['pcl_filtering']['ptsd_threshold']}<br>
            <strong>Buffer:</strong> ±{metadata['pcl_filtering']['buffer']} points
            """)
        
        # Feature types
        if 'feature_types' in metadata:
            self.add_section("Feature Categories", level=3)
            feature_list = "<ul>"
            for feat_type, count in metadata['feature_types'].items():
                feature_list += f"<li>{feat_type}: {count} trials</li>"
            feature_list += "</ul>"
            self.add_text(feature_list)
        
        # Results table
        self.add_section("Model Performance")
        self.add_table(results_df, "Test Set Performance")
        
        # Add visualizations
        self.add_section("Visualizations")
        
        # Model comparison
        model_comp_path = self.output_dir / 'model_comparison.png'
        if model_comp_path.exists():
            self.add_image(model_comp_path, "Model Performance Comparison")
        
        # Task-specific visualizations
        if task_type == 'regression':
            # Scatter plots
            scatter_path = self.output_dir / 'prediction_scatter_plots.png'
            if scatter_path.exists():
                self.add_image(scatter_path, "Prediction Scatter Plots")
            
            # Error distributions
            error_path = self.output_dir / 'error_distributions.png'
            if error_path.exists():
                self.add_image(error_path, "Prediction Error Distributions")
        else:
            # Confusion matrices
            cm_path = self.output_dir / 'confusion_matrices.png'
            if cm_path.exists():
                self.add_image(cm_path, "Confusion Matrices")
            
            cm_pct_path = self.output_dir / 'confusion_matrices_percentage.png'
            if cm_pct_path.exists():
                self.add_image(cm_pct_path, "Normalized Confusion Matrices")
            
            # ROC curves
            roc_path = self.output_dir / 'roc_curves.png'
            if roc_path.exists():
                self.add_image(roc_path, "ROC Curves")
        
        # Feature importance
        feat_imp_path = self.output_dir / 'feature_importance.png'
        if feat_imp_path.exists():
            self.add_section("Feature Importance Analysis")
            self.add_image(feat_imp_path, "Top 20 Most Important Features")
    
    def save_report(self, filename="report.html"):
        """Save the complete HTML report"""
        # HTML template with CSS styling
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>ML Pipeline Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f4f4f4;
                }}
                .header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 30px;
                    text-align: center;
                    margin-bottom: 30px;
                    border-radius: 5px;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 2.5em;
                }}
                .header h2 {{
                    margin: 10px 0 0 0;
                    font-size: 1.5em;
                    font-weight: normal;
                    opacity: 0.9;
                }}
                .timestamp {{
                    margin-top: 15px;
                    font-size: 0.9em;
                    opacity: 0.8;
                }}
                h2 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                    margin-top: 40px;
                }}
                h3 {{
                    color: #34495e;
                    margin-top: 30px;
                }}
                .data-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    background-color: white;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .data-table th {{
                    background-color: #3498db;
                    color: white;
                    padding: 12px;
                    text-align: left;
                    font-weight: bold;
                }}
                .data-table td {{
                    padding: 10px 12px;
                    border-bottom: 1px solid #ddd;
                }}
                .data-table tr:hover {{
                    background-color: #f5f5f5;
                }}
                .data-table tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .image-container {{
                    text-align: center;
                    margin: 20px 0;
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .code-block {{
                    background-color: #2c3e50;
                    color: #ecf0f1;
                    padding: 15px;
                    border-radius: 5px;
                    overflow-x: auto;
                    font-family: 'Courier New', monospace;
                    margin: 20px 0;
                }}
                p {{
                    background-color: white;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 15px 0;
                }}
                strong {{
                    color: #2c3e50;
                }}
                em {{
                    color: #7f8c8d;
                }}
                @media print {{
                    body {{
                        background-color: white;
                    }}
                    .header {{
                        background-color: #2c3e50 !important;
                        -webkit-print-color-adjust: exact;
                    }}
                }}
            </style>
        </head>
        <body>
            {''.join(self.html_content)}
        </body>
        </html>
        """
        
        # Save report
        report_path = self.output_dir / filename
        with open(report_path, 'w') as f:
            f.write(html_template)
        
        print(f"Report saved to: {report_path}")
        print(f"\nTo transfer report, use:")
        print(f"  scp -r {report_path.parent} <destination>")
        print(f"\nAll images and assets are embedded in the HTML file.")
        
        return report_path


def generate_report_for_run(output_dir, report_type=None, use_unified=True):
    """Generate report for a specific run directory"""
    output_path = Path(output_dir)
    
    if use_unified:
        # Use the new unified report generator
        from unified_report_generator import UnifiedReportGenerator
        return generate_unified_report(output_dir, report_type)
    else:
        # Legacy mode
        generator = ReportGenerator(output_dir)
        
        # Auto-detect report type if not specified
        if report_type is None:
            if (output_path / 'kfold_metadata.json').exists():
                report_type = 'kfold'
            else:
                report_type = 'regular'
        
        if report_type == 'kfold':
            # Find k-fold specific files
            metadata_path = output_path / 'kfold_metadata.json'
            
            # Get target from metadata
            with open(metadata_path, 'r') as f:
                target = json.load(f)['target_variable']
            
            results_path = output_path / 'kfold_results_summary.csv'
            detailed_path = output_path / 'kfold_detailed_results.csv'
            
            generator.generate_kfold_report(metadata_path, results_path, detailed_path)
        else:
            # Regular pipeline report
            metadata_path = output_path / 'run_metadata.json'
            
            # Get target from metadata
            with open(metadata_path, 'r') as f:
                target = json.load(f)['target_variable']
            
            results_path = output_path / 'model_results_summary.csv'
            
            generator.generate_regular_report(metadata_path, results_path)
        
        return generator.save_report()

def generate_unified_report(output_dir, report_type=None):
    """Generate unified report with tabs for all content"""
    from unified_report_generator import UnifiedReportGenerator
    
    output_path = Path(output_dir)
    generator = UnifiedReportGenerator(output_dir)
    
    # Auto-detect report type if not specified
    if report_type is None:
        if (output_path / 'kfold_metadata.json').exists():
            report_type = 'kfold'
        else:
            report_type = 'regular'
    
    # Load metadata
    if report_type == 'kfold':
        metadata_path = output_path / 'kfold_metadata.json'
        results_path = output_path / 'kfold_results_summary.csv'
        detailed_path = output_path / 'kfold_detailed_results.csv'
    else:
        metadata_path = output_path / 'run_metadata.json'
        results_path = output_path / 'model_results_summary.csv'
        detailed_path = None
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    results_df = pd.read_csv(results_path)
    
    # Create overview tab
    generator.create_tab('overview', 'Overview')
    task_type = metadata['task_type']
    target = metadata['target_variable']
    
    generator.add_header(
        f"ML Pipeline Report: {target.upper()}",
        f"{task_type.capitalize()} Task - {report_type.upper()}"
    )
    
    # Overview content
    if report_type == 'kfold':
        generator.add_text(f"""
        This report summarizes the results of k-fold cross-validation for {target} prediction.
        The analysis used {metadata['validation_settings']['n_folds']} folds repeated 
        {metadata['validation_settings']['n_iterations']} times for a total of 
        {metadata['validation_settings']['total_evaluations']} evaluations per model.
        """)
    else:
        generator.add_text(f"""
        This report summarizes the machine learning pipeline results for {target} prediction.
        The analysis used a {int(metadata['training_info']['test_size']*100)}% test set 
        with {metadata['training_info']['train_samples']} training samples and 
        {metadata['training_info']['test_samples']} test samples.
        """)
    
    # Create performance tab
    generator.create_tab('performance', 'Model Performance')
    generator.add_section("Performance Summary")
    generator.add_table(results_df, "Model Results", color_code=True)
    
    # Add visualizations
    if task_type == 'classification':
        # Confusion matrices
        cm_path = output_path / ('kfold_confusion_matrices.png' if report_type == 'kfold' else 'confusion_matrices.png')
        if cm_path.exists():
            generator.add_image(cm_path, "Confusion Matrices")
    
    # Create data info tab
    generator.create_tab('data_info', 'Dataset Information')
    generator.add_section("Dataset Details")
    
    generator.add_text(f"""
    <strong>Source:</strong> {metadata['data_info']['source_file']}<br>
    <strong>Total Samples:</strong> {metadata['data_info']['total_samples']}<br>
    <strong>Total Features:</strong> {metadata['data_info']['total_features']}<br>
    <strong>Target Variable:</strong> {target} ({task_type})
    """)
    
    # Create visualizations tab
    generator.create_tab('visualizations', 'Visualizations')
    generator.add_section("Model Comparisons and Analysis")
    
    # Add all visualization images
    viz_files = [
        ('model_comparison.png', 'Model Performance Comparison'),
        ('kfold_model_comparison.png', 'K-Fold Model Comparison'),
        ('roc_curves.png', 'ROC Curves'),
        ('kfold_roc_curves.png', 'K-Fold ROC Curves'),
        ('feature_importance.png', 'Feature Importance'),
        ('prediction_scatter_plots.png', 'Prediction Scatter Plots'),
        ('error_distributions.png', 'Error Distributions'),
        ('kfold_score_distributions.png', 'Score Distributions')
    ]
    
    for filename, title in viz_files:
        img_path = output_path / filename
        if img_path.exists():
            generator.add_image(img_path, title)
    
    # Generate the report
    return generator.generate_html(
        filename="report.html",
        title=f"{target.upper()} - {task_type.capitalize()} Report"
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate HTML report for ML pipeline results')
    parser.add_argument('output_dir', help='Path to the output directory containing results')
    parser.add_argument('--type', choices=['regular', 'kfold'], default='regular',
                        help='Type of report to generate')
    
    args = parser.parse_args()
    generate_report_for_run(args.output_dir, args.type)