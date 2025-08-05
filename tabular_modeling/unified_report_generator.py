#!/usr/bin/env python3
"""
Unified HTML Report Generator for ML Pipeline Results
Generates a single self-contained HTML file with tabs for all content
"""

import base64
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, matthews_corrcoef
import os

class UnifiedReportGenerator:
    """Generate unified HTML reports with all content in tabs"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.tabs = {}  # Store content for each tab
        self.current_tab = None
        
    def create_tab(self, tab_id, tab_name):
        """Create a new tab for content"""
        self.tabs[tab_id] = {
            'name': tab_name,
            'content': []
        }
        self.current_tab = tab_id
        
    def add_to_tab(self, tab_id, content):
        """Add content to specific tab"""
        if tab_id not in self.tabs:
            self.create_tab(tab_id, tab_id)
        self.tabs[tab_id]['content'].append(content)
        
    def add_content(self, content):
        """Add content to current tab"""
        if self.current_tab:
            self.tabs[self.current_tab]['content'].append(content)
            
    def image_to_base64(self, image_path):
        """Convert image file to base64 string for embedding"""
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
        """Add header section"""
        content = f"""
        <div class="header">
            <h1>{title}</h1>
            {f'<h2>{subtitle}</h2>' if subtitle else ''}
            <p class="timestamp">Generated on: {self.timestamp}</p>
        </div>
        """
        self.add_content(content)
        
    def add_section(self, title, level=2):
        """Add section header"""
        self.add_content(f"<h{level}>{title}</h{level}>")
        
    def add_text(self, text, style=""):
        """Add text paragraph"""
        self.add_content(f'<p style="{style}">{text}</p>')
        
    def add_code_block(self, code):
        """Add code block"""
        self.add_content(f'<pre class="code-block">{code}</pre>')
        
    def add_table(self, df, title=None, color_code=False):
        """Add pandas DataFrame as HTML table with optional color coding"""
        if title:
            self.add_content(f'<h3>{title}</h3>')
        
        if color_code and isinstance(df, pd.DataFrame):
            # Apply relative color coding
            html = self._create_color_coded_table(df)
            self.add_content(html)
        else:
            # Regular table
            if isinstance(df, pd.DataFrame):
                # Format numeric columns
                for col in df.select_dtypes(include=[np.number]).columns:
                    if 'Mean' in col or 'Std' in col or 'Score' in col or 'R²' in col:
                        df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "")
                    elif 'RMSE' in col or 'MAE' in col or 'MSE' in col:
                        df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
            
            table_html = df.to_html(index=False, classes='data-table')
            self.add_content(table_html)
            
    def _create_color_coded_table(self, df):
        """Create table with relative color coding for metrics"""
        # Identify metric columns
        metric_columns = {
            'AUC', 'AUC Mean', 'Accuracy', 'Accuracy Mean', 'Test Accuracy',
            'MCC', 'MCC Mean', 'R²', 'R² Mean', 'Test R²',
            'F1 Score', 'F1 Mean', 'Precision', 'Precision Mean', 
            'Recall', 'Recall Mean'
        }
        
        html = '<table class="data-table">\n<thead>\n<tr>\n'
        
        # Headers
        for col in df.columns:
            html += f'<th>{col}</th>\n'
        html += '</tr>\n</thead>\n<tbody>\n'
        
        # Pre-calculate values for relative coloring
        column_values = {}
        for col in df.columns:
            if col in metric_columns:
                try:
                    numeric_values = pd.to_numeric(df[col], errors='coerce').dropna()
                    if len(numeric_values) > 0:
                        column_values[col] = numeric_values.tolist()
                except:
                    pass
        
        # Rows with color coding
        for idx, row in df.iterrows():
            html += '<tr>\n'
            for col in df.columns:
                value = row[col]
                cell_class = ''
                
                # Apply relative color coding
                if col in column_values:
                    try:
                        numeric_value = float(value)
                        values = column_values[col]
                        rank = sum(1 for v in sorted(values) if v <= numeric_value) / len(values)
                        
                        if rank >= 0.8:
                            cell_class = 'metric-excellent'
                        elif rank >= 0.6:
                            cell_class = 'metric-good'
                        elif rank >= 0.4:
                            cell_class = 'metric-fair'
                        elif rank >= 0.2:
                            cell_class = 'metric-poor'
                        else:
                            cell_class = 'metric-very-poor'
                    except:
                        pass
                
                if cell_class:
                    html += f'<td class="{cell_class}">{value}</td>\n'
                else:
                    html += f'<td>{value}</td>\n'
            html += '</tr>\n'
        
        html += '</tbody>\n</table>\n'
        return html
        
    def add_image(self, image_path, title=None, width="100%"):
        """Add image from file"""
        img_base64 = self.image_to_base64(image_path)
        if img_base64:
            if title:
                self.add_content(f'<h3>{title}</h3>')
            self.add_content(f'''
            <div class="image-container">
                <img src="data:image/png;base64,{img_base64}" style="width: {width}; max-width: 100%;">
            </div>
            ''')
            
    def add_figure(self, fig, title=None, width="100%"):
        """Add matplotlib figure directly"""
        img_base64 = self.figure_to_base64(fig)
        if title:
            self.add_content(f'<h3>{title}</h3>')
        self.add_content(f'''
        <div class="image-container">
            <img src="data:image/png;base64,{img_base64}" style="width: {width}; max-width: 100%;">
        </div>
        ''')
        
    def add_confusion_matrix_table(self, cm, model_name, classes=['Negative', 'Positive']):
        """Add confusion matrix as HTML table with percentages and raw counts"""
        total = cm.sum()
        
        html = f'<div class="confusion-matrix">\n'
        html += f'<h4>{model_name}</h4>\n'
        html += '<table>\n'
        
        # Header row
        html += '<tr>'
        html += '<td class="cm-header"></td>'
        for pred_class in classes:
            html += f'<td class="cm-header">Predicted {pred_class}</td>'
        html += '</tr>\n'
        
        # Data rows
        for i, true_class in enumerate(classes):
            html += f'<tr><td class="cm-header">Actual {true_class}</td>'
            for j in range(len(classes)):
                count = cm[i, j]
                percentage = (count / total) * 100
                cell_color = '#d4edda' if i == j else '#f8d7da'
                html += f'<td style="background-color: {cell_color}; text-align: center; font-weight: bold;">'
                html += f'{percentage:.1f}% ({count})</td>'
            html += '</tr>\n'
        
        html += '</table>\n'
        
        # Add metrics
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / total
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            mcc = matthews_corrcoef([0]*tn + [0]*fp + [1]*fn + [1]*tp,
                                  [0]*tn + [1]*fp + [0]*fn + [1]*tp)
            
            html += '<div class="cm-metrics">'
            html += f'<strong>Accuracy:</strong> {accuracy:.3f} | '
            html += f'<strong>Precision:</strong> {precision:.3f} | '
            html += f'<strong>Recall:</strong> {recall:.3f} | '
            html += f'<strong>F1:</strong> {f1:.3f} | '
            html += f'<strong>MCC:</strong> {mcc:.3f}'
            html += '</div>'
        
        html += '</div>\n'
        self.add_content(html)
        
    def generate_html(self, filename="report.html", title="ML Pipeline Report"):
        """Generate the complete HTML report with tabs"""
        
        # Build tab navigation
        tab_nav = '<div class="tab-nav">\n'
        for i, (tab_id, tab_info) in enumerate(self.tabs.items()):
            active_class = 'active' if i == 0 else ''
            tab_nav += f'<button class="tab-button {active_class}" onclick="showTab(\'{tab_id}\')">{tab_info["name"]}</button>\n'
        tab_nav += '</div>\n'
        
        # Build tab contents
        tab_contents = ''
        for i, (tab_id, tab_info) in enumerate(self.tabs.items()):
            active_class = 'active' if i == 0 else ''
            tab_contents += f'<div id="{tab_id}" class="tab-content {active_class}">\n'
            tab_contents += '\n'.join(tab_info['content'])
            tab_contents += '</div>\n'
        
        # Complete HTML with CSS and JavaScript
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 30px;
            text-align: center;
            border-radius: 5px;
            margin-bottom: 20px;
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
        .tab-nav {{
            display: flex;
            background-color: white;
            border-radius: 5px 5px 0 0;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 0;
        }}
        .tab-button {{
            flex: 1;
            padding: 15px 20px;
            border: none;
            background-color: #ecf0f1;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: background-color 0.3s;
        }}
        .tab-button:hover {{
            background-color: #bdc3c7;
        }}
        .tab-button.active {{
            background-color: #3498db;
            color: white;
        }}
        .tab-content {{
            background-color: white;
            padding: 30px;
            border-radius: 0 0 5px 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: none;
            min-height: 500px;
        }}
        .tab-content.active {{
            display: block;
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
        /* Color coding for metrics */
        .metric-excellent {{ background-color: #2ecc71; color: white; }}
        .metric-good {{ background-color: #27ae60; color: white; }}
        .metric-fair {{ background-color: #f39c12; color: white; }}
        .metric-poor {{ background-color: #e74c3c; color: white; }}
        .metric-very-poor {{ background-color: #c0392b; color: white; }}
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
            border-collapse: collapse;
        }}
        .confusion-matrix td {{
            padding: 12px;
            text-align: center;
            min-width: 120px;
        }}
        .cm-header {{
            background-color: #34495e !important;
            color: white !important;
            font-weight: bold;
        }}
        .cm-metrics {{
            margin-top: 15px;
            text-align: center;
            font-size: 14px;
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
    </style>
    <script>
        function showTab(tabId) {{
            // Hide all tab contents
            var contents = document.getElementsByClassName('tab-content');
            for (var i = 0; i < contents.length; i++) {{
                contents[i].classList.remove('active');
            }}
            
            // Remove active class from all buttons
            var buttons = document.getElementsByClassName('tab-button');
            for (var i = 0; i < buttons.length; i++) {{
                buttons[i].classList.remove('active');
            }}
            
            // Show selected tab
            document.getElementById(tabId).classList.add('active');
            
            // Activate corresponding button
            var allButtons = document.querySelectorAll('.tab-button');
            for (var i = 0; i < allButtons.length; i++) {{
                if (allButtons[i].getAttribute('onclick').includes(tabId)) {{
                    allButtons[i].classList.add('active');
                }}
            }}
        }}
    </script>
</head>
<body>
    <div class="container">
        {tab_nav}
        {tab_contents}
    </div>
</body>
</html>
"""
        
        # Save report
        report_path = self.output_dir / filename
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        print(f"Report saved to: {report_path}")
        print(f"\nThis is a self-contained HTML file with all images embedded.")
        print(f"To transfer: scp {report_path} <destination>")
        
        return report_path