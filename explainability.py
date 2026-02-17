import pandas as pd
import numpy as np
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
except SystemError:
    SHAP_AVAILABLE = False
except Exception:
    SHAP_AVAILABLE = False

import plotly.express as px
import plotly.io as pio
from datetime import datetime
from typing import Dict, Any, List, Optional
import json

class Explainer:
    def __init__(self):
        # Initialize any global state if needed
        pass

    def compute_shap(self, model: Any, X: pd.DataFrame, top_k: int = 10) -> Dict[str, Any]:
        """
        Computes SHAP values for tree-based models (LGBM, XGB).
        Returns top_k features and their contribution for the *last* instance in X.
        """
        if not SHAP_AVAILABLE:
            # Fallback for environments where SHAP/Numba is broken
            return {
                "shap_values": {"error": "SHAP library not available"},
                "top_features": {"error": "SHAP library not available"}
            }

        try:
            # TreeExplainer is fast for trees
            explainer = shap.TreeExplainer(model)
            
            # Compute SHAP values
            # shap_values shape: (n_samples, n_features)
            # For multi-class, it might be a list. Assuming binary/regression here.
            # If Model is sklearn wrapper, accessing .booster_ might be better, 
            # but TreeExplainer handles sklearn wrappers usually.
            
            # Using the last instance for "Real-time" explanation
            X_last = X.iloc[[-1]] 
            shap_values = explainer.shap_values(X_last)
            
            # Handle list output (for some LGBM versions binary classification returns list of 2)
            if isinstance(shap_values, list):
                # Class 1 (Positive)
                shap_values = shap_values[1]
            
            # It might return (1, n_features)
            if len(shap_values.shape) > 1:
                vals = shap_values[0]
            else:
                vals = shap_values
                
            # Create dict mapping
            feature_names = X.columns.tolist()
            feature_impact = dict(zip(feature_names, vals))
            
            # Sort by absolute impact
            sorted_features = sorted(feature_impact.items(), key=lambda item: abs(item[1]), reverse=True)
            top_features = dict(sorted_features[:top_k])
            
            return {
                "shap_values": feature_impact,
                "top_features": top_features
            }
        except Exception as e:
            print(f"SHAP computation failed: {e}")
            return {"error": str(e)}

    def generate_daily_report(self, logs: List[Dict[str, Any]], output_path: str = "daily_report.html"):
        """
        Generates an HTML report from explainability logs.
        """
        if not logs:
            return
            
        # Aggregate Top Features
        all_top_features = []
        for log in logs:
            if 'top_features' in log and log['top_features']:
                all_top_features.extend(log['top_features'].keys())
                
        # Count frequency
        from collections import Counter
        feat_counts = Counter(all_top_features)
        df_feat = pd.DataFrame(feat_counts.items(), columns=['Feature', 'Count']).sort_values('Count', ascending=False).head(20)
        
        # Plot
        fig = px.bar(df_feat, x='Count', y='Feature', orientation='h', title="Top Driver Features (Daily Aggregate)")
        chart_html = pio.to_html(fig, full_html=False)
        
        # Generate HTML
        html_content = f"""
        <html>
        <head><title>Holographic Glenn - Daily Explainability Report</title></head>
        <body>
            <h1>Daily Explainability Report</h1>
            <p>Generated at: {datetime.now()}</p>
            <hr>
            <h2>Top Feature Drivers</h2>
            {chart_html}
            <hr>
            <h2>Summary</h2>
            <p>Total Predictions Analyzed: {len(logs)}</p>
        </body>
        </html>
        """
        
        with open(output_path, "w") as f:
            f.write(html_content)
        print(f"Report saved to {output_path}")

# Mock Model for testing/usage if real model not passed
class MockModel:
    def predict(self, X):
        return np.random.rand(len(X))
