#!/usr/bin/env python3
"""
Generate combined HTML report with tabs for all ML pipeline variations
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import matthews_corrcoef, confusion_matrix
import shutil
import os

class CombinedReportGenerator:
    """Generate combined HTML report with tabs"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.assets_copied = []
        
    def load_variation_data(self, run_folder):
        """Load data from a specific run"""
        run_path = Path(run_folder)
        data = {}
        
        # Load metadata - check both possible locations
        kfold_metadata_path = run_path / 'kfold_metadata.json'
        regular_metadata_path = run_path / 'run_metadata.json'
        
        if kfold_metadata_path.exists():
            with open(kfold_metadata_path, 'r') as f:
                data['metadata'] = json.load(f)
            data['type'] = 'kfold'
        elif regular_metadata_path.exists():
            with open(regular_metadata_path, 'r') as f:
                data['metadata'] = json.load(f)
            data['type'] = 'regular'
        else:
            raise FileNotFoundError(f"Neither kfold_metadata.json nor run_metadata.json found in {run_path}")
        
        # Load results
        if data['type'] == 'kfold':
            results_file = run_path / 'kfold_results_summary.csv'
        else:
            results_file = run_path / 'model_results_summary.csv'
        
        if results_file.exists():
            data['results'] = pd.read_csv(results_file)
        
        # Store path for loading images
        data['path'] = run_path
        
        return data
    
    def _create_comparison_figure(self, all_data):
        """Create comparison figure for unified report"""
        return self.create_comparison_plot(all_data, return_figure=True)
        
    def _generate_insights_html(self, all_data):
        """Generate insights HTML for unified report"""
        return self._generate_insights(all_data)
    
    def create_comparison_plot(self, all_data, return_figure=False):
        """Create comparison plot across all variations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        variations = ['Regular', 'Regular + PCL Filter', 'K-Fold', 'K-Fold + PCL Filter']
        
        for idx, (name, data) in enumerate(all_data.items()):
            if idx < len(axes) and 'results' in data:
                ax = axes[idx]
                results = data['results']
                
                # Get metric based on type
                if data['type'] == 'kfold':
                    if 'AUC Mean' in results.columns:
                        metric_col = 'AUC Mean'
                        ylabel = 'AUC Score'
                    else:
                        metric_col = 'R² Mean'
                        ylabel = 'R² Score'
                else:
                    if 'AUC' in results.columns:
                        metric_col = 'AUC'
                        ylabel = 'AUC Score'
                    else:
                        metric_col = 'Test R²'
                        ylabel = 'R² Score'
                
                # Plot
                models = results['Model'].values
                scores = results[metric_col].values
                
                bars = ax.bar(range(len(models)), scores, alpha=0.7)
                ax.set_xticks(range(len(models)))
                ax.set_xticklabels(models, rotation=45, ha='right')
                ax.set_ylabel(ylabel)
                ax.set_title(variations[idx])
                ax.grid(True, alpha=0.3)
                
                # Color bars based on performance
                for bar, score in zip(bars, scores):
                    if score < 0.5:
                        bar.set_color('red')
                    elif score < 0.6:
                        bar.set_color('orange')
                    else:
                        bar.set_color('green')
        
        plt.suptitle('Model Performance Across All Variations', fontsize=16)
        plt.tight_layout()
        
        if return_figure:
            return fig
        else:
            # Convert to base64
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode()
            buf.close()
            plt.close()
            
            return img_base64
    
    def copy_assets_from_runs(self, run_folders):
        """Copy all image assets from run folders to output directory"""
        assets_dir = self.output_dir / 'assets'
        assets_dir.mkdir(exist_ok=True)
        
        for name, folder_path in run_folders.items():
            folder = Path(folder_path)
            if folder.exists():
                # Copy all PNG files
                for png_file in folder.glob('*.png'):
                    dest_name = f"{name}_{png_file.name}"
                    dest_path = assets_dir / dest_name
                    shutil.copy2(png_file, dest_path)
                    self.assets_copied.append((png_file, dest_path))
                
                # Copy report.html if exists
                report_file = folder / 'report.html'
                if report_file.exists():
                    dest_path = self.output_dir / f"{name}_report.html"
                    shutil.copy2(report_file, dest_path)
    
    def generate_combined_report(self, run_folders, single_file=True):
        """Generate the combined HTML report"""
        if single_file:
            # Use unified report generator for single file output
            from unified_report_generator import UnifiedReportGenerator
            unified_gen = UnifiedReportGenerator(self.output_dir)
            
            # Load data from all runs
            all_data = {}
            for name, folder in run_folders.items():
                try:
                    all_data[name] = self.load_variation_data(folder)
                except Exception as e:
                    print(f"Error loading {name}: {e}")
            
            # Create overview tab with comparison
            unified_gen.create_tab('overview', 'Overview')
            unified_gen.add_header("Combined ML Pipeline Report", "All Variations Comparison")
            
            # Add comparison plot
            comparison_fig = self._create_comparison_figure(all_data)
            unified_gen.add_figure(comparison_fig, "Model Performance Across All Variations")
            
            # Add insights
            unified_gen.add_section("Key Insights")
            insights_html = self._generate_insights_html(all_data)
            unified_gen.add_content(insights_html)
            
            # Create a tab for each variation
            for name, data in all_data.items():
                tab_name = name.replace('_', ' ').title()
                unified_gen.create_tab(name, tab_name)
                
                # Add metadata summary
                if 'metadata' in data:
                    meta = data['metadata']
                    unified_gen.add_section("Configuration")
                    unified_gen.add_text(f"""
                    <strong>Target:</strong> {meta['target_variable']}<br>
                    <strong>Task Type:</strong> {meta['task_type']}<br>
                    <strong>Samples:</strong> {meta['data_info']['total_samples']}<br>
                    <strong>Features:</strong> {meta['data_info']['total_features']}
                    """)
                    
                    if data['type'] == 'kfold' and 'validation_settings' in meta:
                        val = meta['validation_settings']
                        unified_gen.add_text(f"""
                        <strong>K-Fold Settings:</strong><br>
                        - Folds: {val['n_folds']}<br>
                        - Iterations: {val['n_iterations']}<br>
                        - Total Evaluations: {val['total_evaluations']}
                        """)
                
                # Add results table
                if 'results' in data:
                    unified_gen.add_section("Model Performance")
                    unified_gen.add_table(data['results'], color_code=True)
                
                # Add visualizations from the run
                if 'path' in data:
                    unified_gen.add_section("Visualizations")
                    viz_files = [
                        ('model_comparison.png', 'Model Comparison'),
                        ('kfold_model_comparison.png', 'Model Comparison'),
                        ('confusion_matrices.png', 'Confusion Matrices'),
                        ('kfold_confusion_matrices.png', 'Confusion Matrices'),
                        ('roc_curves.png', 'ROC Curves'),
                        ('kfold_roc_curves.png', 'ROC Curves'),
                        ('feature_importance.png', 'Feature Importance')
                    ]
                    
                    for filename, title in viz_files:
                        img_path = data['path'] / filename
                        if img_path.exists():
                            unified_gen.add_image(img_path, title)
            
            # Generate single HTML file
            report_path = unified_gen.generate_html(
                filename="combined_report.html",
                title="Combined ML Pipeline Analysis"
            )
            
            print(f"\n{'='*60}")
            print("Report generation complete!")
            print(f"{'='*60}")
            print(f"\nSingle HTML file saved to: {report_path}")
            print(f"\nTo transfer:")
            print(f"  scp {report_path} <user>@<host>:<destination>")
            print(f"\nThe report is completely self-contained with all images embedded.")
            print(f"{'='*60}\n")
            
            return report_path
            
        else:
            # Original multi-file approach
            self.copy_assets_from_runs(run_folders)
            
            # Load data from all runs
            all_data = {}
            for name, folder in run_folders.items():
                try:
                    all_data[name] = self.load_variation_data(folder)
                except Exception as e:
                    print(f"Error loading {name}: {e}")
            
            # Create comparison plot
            comparison_plot = self.create_comparison_plot(all_data)
        
        # Generate HTML
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Combined ML Pipeline Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .timestamp {{
            margin-top: 15px;
            font-size: 0.9em;
            opacity: 0.8;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        .tabs {{
            display: flex;
            background-color: white;
            border-radius: 5px 5px 0 0;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .tab {{
            flex: 1;
            padding: 15px 20px;
            text-align: center;
            cursor: pointer;
            background-color: #ecf0f1;
            border: none;
            outline: none;
            transition: background-color 0.3s;
            font-size: 16px;
            font-weight: bold;
        }}
        .tab:hover {{
            background-color: #bdc3c7;
        }}
        .tab.active {{
            background-color: #3498db;
            color: white;
        }}
        .tab-content {{
            background-color: white;
            padding: 30px;
            border-radius: 0 0 5px 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        .summary-card h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
        }}
        .metric-label {{
            font-weight: bold;
        }}
        .metric-value {{
            color: #3498db;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: white;
        }}
        th {{
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        /* Color coding for performance metrics */
        .metric-excellent {{ background-color: #2ecc71; color: white; }}
        .metric-good {{ background-color: #27ae60; color: white; }}
        .metric-fair {{ background-color: #f39c12; color: white; }}
        .metric-poor {{ background-color: #e74c3c; color: white; }}
        .metric-very-poor {{ background-color: #c0392b; color: white; }}
        /* Confusion matrix styling */
        .confusion-matrix {{
            margin: 20px 0;
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
        }}
        .confusion-matrix table {{
            margin: auto;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .confusion-matrix td {{
            text-align: center;
            min-width: 100px;
            font-weight: bold;
        }}
        .cm-header {{
            background-color: #34495e !important;
            color: white !important;
        }}
        .image-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .comparison-section {{
            background-color: #f8f9fa;
            padding: 30px;
            margin: 30px 0;
            border-radius: 5px;
        }}
        .warning {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .success {{
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        iframe {{
            width: 100%;
            height: 800px;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin: 20px 0;
        }}
    </style>
    <script>
        function showTab(tabName) {{
            // Hide all tab contents
            var contents = document.getElementsByClassName('tab-content');
            for (var i = 0; i < contents.length; i++) {{
                contents[i].classList.remove('active');
            }}
            
            // Remove active class from all tabs
            var tabs = document.getElementsByClassName('tab');
            for (var i = 0; i < tabs.length; i++) {{
                tabs[i].classList.remove('active');
            }}
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            document.getElementById(tabName + '-tab').classList.add('active');
        }}
    </script>
</head>
<body>
    <div class="header">
        <h1>Combined ML Pipeline Report</h1>
        <h2>All Variations Comparison</h2>
        <p class="timestamp">Generated on: {self.timestamp}</p>
    </div>
    
    <div class="container">
        <div class="comparison-section">
            <h2>Overall Comparison</h2>
            <div class="image-container">
                <img src="data:image/png;base64,{comparison_plot}" style="max-width: 100%;">
            </div>
"""

        # Add key insights
        html_content += self._generate_insights(all_data)
        
        # Add tabs
        html_content += """
        <div class="tabs">
            <button class="tab active" id="regular-tab" onclick="showTab('regular')">Regular</button>
            <button class="tab" id="regular_filtered-tab" onclick="showTab('regular_filtered')">Regular + PCL Filter</button>
            <button class="tab" id="kfold-tab" onclick="showTab('kfold')">K-Fold</button>
            <button class="tab" id="kfold_filtered-tab" onclick="showTab('kfold_filtered')">K-Fold + PCL Filter</button>
        </div>
"""

        # Add tab contents
        tab_names = {
            'regular': 'Regular (Single Train/Test Split)',
            'regular_filtered': 'Regular with PCL Filtering',
            'kfold': 'K-Fold Cross-Validation',
            'kfold_filtered': 'K-Fold with PCL Filtering'
        }
        
        for idx, (key, title) in enumerate(tab_names.items()):
            is_active = 'active' if idx == 0 else ''
            html_content += f'<div id="{key}" class="tab-content {is_active}">'
            
            if key in all_data:
                html_content += self._generate_variation_content(key, title, all_data[key])
            else:
                html_content += f'<p>No data available for {title}</p>'
            
            html_content += '</div>'
        
        html_content += """
        </div>
    </div>
</body>
</html>
"""
        
        # Save report
        report_path = self.output_dir / 'combined_report.html'
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"Combined report saved to: {report_path}")
        
        # Save metadata
        metadata = {
            'report_type': 'combined',
            'timestamp': self.timestamp,
            'variations': list(run_folders.keys()),
            'run_folders': run_folders,
            'assets_copied': len(self.assets_copied)
        }
        
        with open(self.output_dir / 'combined_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Print transfer instructions
        print(f"\n{'='*60}")
        print("Report generation complete!")
        print(f"{'='*60}")
        print(f"\nAll files saved to: {self.output_dir}")
        print(f"- Main report: combined_report.html")
        print(f"- Detailed reports: {len([f for f in self.output_dir.glob('*_report.html')])} files")
        print(f"- Image assets: {len(self.assets_copied)} files in assets/")
        print(f"\nTo transfer the entire report package:")
        print(f"  scp -r {self.output_dir} <user>@<host>:<destination>")
        print(f"\nThe report folder is self-contained with all necessary files.")
        print(f"{'='*60}\n")
    
    def _generate_insights(self, all_data):
        """Generate insights comparing all variations"""
        insights = '<div class="insights">\n<h3>Key Insights</h3>\n'
        
        # Compare sample sizes
        sample_sizes = {}
        for name, data in all_data.items():
            if 'metadata' in data:
                samples = data['metadata']['data_info']['total_samples']
                sample_sizes[name] = samples
        
        if sample_sizes:
            insights += '<h4>Sample Sizes:</h4><ul>'
            for name, size in sample_sizes.items():
                label = name.replace('_', ' ').title()
                insights += f'<li>{label}: {size} samples</li>'
            insights += '</ul>'
        
        # Find best performing model across all variations
        best_performances = []
        for name, data in all_data.items():
            if 'results' in data:
                results = data['results']
                if data['type'] == 'kfold':
                    if 'AUC Mean' in results.columns:
                        best_idx = results['AUC Mean'].idxmax()
                        best_model = results.loc[best_idx, 'Model']
                        best_score = results.loc[best_idx, 'AUC Mean']
                        metric = 'AUC'
                    else:
                        continue
                else:
                    if 'AUC' in results.columns:
                        best_idx = results['AUC'].idxmax()
                        best_model = results.loc[best_idx, 'Model']
                        best_score = results.loc[best_idx, 'AUC']
                        metric = 'AUC'
                    else:
                        continue
                
                best_performances.append({
                    'variation': name.replace('_', ' ').title(),
                    'model': best_model,
                    'score': best_score,
                    'metric': metric
                })
        
        if best_performances:
            insights += '<h4>Best Performing Models:</h4><ul>'
            for perf in sorted(best_performances, key=lambda x: x['score'], reverse=True):
                insights += f"<li>{perf['variation']}: {perf['model']} ({perf['metric']} = {perf['score']:.4f})</li>"
            insights += '</ul>'
        
        # Add warnings about performance
        all_poor = all(p['score'] < 0.6 for p in best_performances if p['metric'] == 'AUC')
        if all_poor:
            insights += '''
            <div class="warning">
                <strong>⚠️ Performance Warning:</strong> All models show AUC scores close to 0.5 (random chance), 
                indicating they struggle to distinguish between PTSD positive and negative cases. This suggests:
                <ul>
                    <li>The current features may not be sufficiently predictive</li>
                    <li>The relationship between eye-tracking data and PTSD may be more complex</li>
                    <li>Additional feature engineering or data collection may be needed</li>
                </ul>
            </div>
            '''
        
        insights += '</div>'
        return insights
    
    def _get_relative_color_class(self, value, values_list):
        """Get color class based on relative position in the list of values"""
        try:
            # Convert to float and filter out non-numeric values
            numeric_values = []
            for v in values_list:
                try:
                    numeric_values.append(float(v))
                except:
                    pass
            
            if not numeric_values or len(numeric_values) < 2:
                return ''
            
            value_float = float(value)
            
            # Calculate percentile rank
            sorted_values = sorted(numeric_values)
            rank = sum(1 for v in sorted_values if v <= value_float) / len(sorted_values)
            
            # Assign color based on percentile
            if rank >= 0.8:  # Top 20%
                return 'metric-excellent'
            elif rank >= 0.6:  # Top 40%
                return 'metric-good'
            elif rank >= 0.4:  # Middle 20%
                return 'metric-fair'
            elif rank >= 0.2:  # Bottom 40%
                return 'metric-poor'
            else:  # Bottom 20%
                return 'metric-very-poor'
        except:
            return ''
    
    def _format_results_table_with_colors(self, results_df, data_type):
        """Format results table with color coding"""
        html = '<table class="data-table">\n<thead>\n<tr>\n'
        
        # Add headers
        for col in results_df.columns:
            html += f'<th>{col}</th>\n'
        html += '</tr>\n</thead>\n<tbody>\n'
        
        # Pre-calculate values for each metric column for relative coloring
        metric_columns = {
            'AUC': [], 'AUC Mean': [], 'Accuracy': [], 'Accuracy Mean': [],
            'MCC': [], 'MCC Mean': [], 'R²': [], 'R² Mean': [], 'Test R²': [],
            'Test Accuracy': [], 'F1 Score': [], 'F1 Mean': [],
            'Precision': [], 'Precision Mean': [], 'Recall': [], 'Recall Mean': []
        }
        
        # Collect all values for each metric
        for col in metric_columns.keys():
            if col in results_df.columns:
                metric_columns[col] = results_df[col].tolist()
        
        # Add rows with color coding
        for idx, row in results_df.iterrows():
            html += '<tr>\n'
            for col in results_df.columns:
                value = row[col]
                cell_class = ''
                
                # Apply relative color coding to metric columns
                if col in metric_columns and metric_columns[col]:
                    cell_class = self._get_relative_color_class(value, metric_columns[col])
                
                if cell_class:
                    html += f'<td class="{cell_class}">{value}</td>\n'
                else:
                    html += f'<td>{value}</td>\n'
            html += '</tr>\n'
        
        html += '</tbody>\n</table>\n'
        return html
    
    def _load_confusion_matrices(self, run_path, model_names):
        """Load confusion matrix data from the run directory"""
        confusion_matrices = {}
        
        # Try to load from detailed results or metadata
        if (run_path / 'kfold_detailed_results.csv').exists():
            # For k-fold, we need to aggregate confusion matrices
            detailed_df = pd.read_csv(run_path / 'kfold_detailed_results.csv')
            
            for model in model_names:
                model_data = detailed_df[detailed_df['Model'] == model]
                if not model_data.empty:
                    # Simple placeholder - in reality we'd need to aggregate properly
                    # For now, return None to indicate data not available
                    confusion_matrices[model] = None
        
        return confusion_matrices
    
    def _generate_confusion_matrix_html(self, cm, model_name):
        """Generate HTML for a confusion matrix"""
        if cm is None:
            return ""
            
        total = cm.sum()
        html = f'<div class="confusion-matrix">\n'
        html += f'<h4>{model_name} - Confusion Matrix</h4>\n'
        html += '<table>\n'
        html += '<tr><td class="cm-header"></td><td class="cm-header">Predicted Negative</td><td class="cm-header">Predicted Positive</td></tr>\n'
        
        # Calculate percentages
        tn, fp, fn, tp = cm.ravel()
        
        html += f'<tr><td class="cm-header">Actual Negative</td>'
        html += f'<td>{tn/total*100:.1f}% ({tn})</td>'
        html += f'<td>{fp/total*100:.1f}% ({fp})</td></tr>\n'
        
        html += f'<tr><td class="cm-header">Actual Positive</td>'
        html += f'<td>{fn/total*100:.1f}% ({fn})</td>'
        html += f'<td>{tp/total*100:.1f}% ({tp})</td></tr>\n'
        
        html += '</table>\n</div>\n'
        return html

    def _generate_variation_content(self, key, title, data):
        """Generate content for a specific variation tab"""
        content = f'<h2>{title}</h2>\n'
        
        # Add metadata summary
        if 'metadata' in data:
            meta = data['metadata']
            content += '<div class="summary-grid">\n'
            
            # Data info card
            content += '<div class="summary-card">\n<h3>Dataset Information</h3>\n'
            content += f'<div class="metric"><span class="metric-label">Samples:</span><span class="metric-value">{meta["data_info"]["total_samples"]}</span></div>\n'
            content += f'<div class="metric"><span class="metric-label">Features:</span><span class="metric-value">{meta["data_info"]["total_features"]}</span></div>\n'
            content += f'<div class="metric"><span class="metric-label">Target:</span><span class="metric-value">{meta["target_variable"]}</span></div>\n'
            
            if 'pcl_filtering' in meta and meta['pcl_filtering']['remove_intermediate']:
                content += f'<div class="metric"><span class="metric-label">PCL Filter:</span><span class="metric-value">Excluded {meta["pcl_filtering"]["excluded_range"]}</span></div>\n'
            
            content += '</div>\n'
            
            # Pipeline info card
            content += '<div class="summary-card">\n<h3>Pipeline Information</h3>\n'
            content += f'<div class="metric"><span class="metric-label">Type:</span><span class="metric-value">{data["type"].upper()}</span></div>\n'
            
            if data['type'] == 'kfold' and 'validation_settings' in meta:
                val = meta['validation_settings']
                content += f'<div class="metric"><span class="metric-label">Folds:</span><span class="metric-value">{val["n_folds"]}</span></div>\n'
                content += f'<div class="metric"><span class="metric-label">Iterations:</span><span class="metric-value">{val["n_iterations"]}</span></div>\n'
                content += f'<div class="metric"><span class="metric-label">Total Evaluations:</span><span class="metric-value">{val["total_evaluations"]}</span></div>\n'
            
            content += '</div>\n</div>\n'
        
        # Add results table with color coding
        if 'results' in data:
            content += '<h3>Model Performance</h3>\n'
            
            # First, add MCC to the results if it's not there
            results_df = data['results'].copy()
            
            # Check if this is a classification task and MCC is missing
            is_classification = 'AUC' in results_df.columns or 'AUC Mean' in results_df.columns
            
            if is_classification and 'MCC' not in results_df.columns and 'MCC Mean' not in results_df.columns:
                # For now, we'll add placeholder MCC values (in a real scenario, we'd calculate from confusion matrices)
                # This is a temporary solution - proper MCC calculation would require access to predictions
                if 'AUC Mean' in results_df.columns:
                    # For k-fold results, add MCC Mean and MCC Std
                    results_df['MCC Mean'] = 'N/A'
                    results_df['MCC Std'] = 'N/A'
                else:
                    # For regular results, add MCC
                    results_df['MCC'] = 'N/A'
            
            # Format table with color coding
            content += self._format_results_table_with_colors(results_df, data['type'])
            
            # Add confusion matrices for classification tasks
            if is_classification and 'path' in data:
                content += '<h3>Confusion Matrices</h3>\n'
                content += '<p><em>Note: Confusion matrices are available in the detailed report. ' + \
                          'Click the link below to view full confusion matrices with percentages and raw counts.</em></p>\n'
        
        # Add link to full report
        if 'path' in data:
            report_file = data['path'] / 'report.html'
            if report_file.exists():
                content += f'<h3>Detailed Report</h3>\n'
                content += f'<p>View the <a href="{key}_report.html" target="_blank">full detailed report</a> for this variation.</p>\n'
        
        return content

if __name__ == "__main__":
    # Test the generator
    import argparse
    import subprocess
    
    parser = argparse.ArgumentParser(description='Generate combined report from multiple runs')
    parser.add_argument('run_folders', nargs='+', help='Folders containing run results')
    parser.add_argument('--scp', action='store_true', 
                        help='Automatically SCP the report to local computer after generation')
    parser.add_argument('--scp-dest', default='/Users/michaelzakariaie/Desktop',
                        help='Destination path for SCP (default: /Users/michaelzakariaie/Desktop)')
    
    args = parser.parse_args()
    
    # Create mapping
    run_folders = {}
    for i, folder in enumerate(args.run_folders):
        if 'kfold' in folder:
            if 'filtered' in folder or '25_41' in folder:
                run_folders['kfold_filtered'] = folder
            else:
                run_folders['kfold'] = folder
        else:
            if 'filtered' in folder or '25_41' in folder:
                run_folders['regular_filtered'] = folder
            else:
                run_folders['regular'] = folder
    
    from run_counter import get_next_run_folder
    output_path, _ = get_next_run_folder('ml_output')
    output_path.mkdir(parents=True, exist_ok=True)
    
    generator = CombinedReportGenerator(output_path)
    generator.generate_combined_report(run_folders)
    
    # SCP if requested
    if args.scp:
        # Get the current machine's IP (assuming it's 192.168.0.119 based on your example)
        source_path = output_path.absolute()
        dest_path = args.scp_dest
        
        print(f"\n{'='*60}")
        print("Starting SCP transfer...")
        print(f"{'='*60}")
        
        # Build SCP command - from remote to local requires running from local machine
        # So we'll provide the command for the user to run locally
        scp_command = f"scp -r michael@192.168.0.119:{source_path} {dest_path}"
        
        print(f"\nTo transfer the report to your local machine, run this command from your local terminal:")
        print(f"\n  {scp_command}\n")
        
        # Alternatively, if you have SSH keys set up in reverse, we could try:
        # This would work if the remote server can SSH back to your local machine
        try:
            # Try to get the SSH client IP from environment
            ssh_client = os.environ.get('SSH_CLIENT', '').split()[0]
            if ssh_client:
                print(f"Detected your local IP: {ssh_client}")
                print("If you have reverse SSH set up, you could also try:")
                print(f"  scp -r {source_path} michael@{ssh_client}:{dest_path}")
        except:
            pass
        
        print(f"\nReport folder ready for transfer: {output_path.name}/")
        print(f"{'='*60}\n")