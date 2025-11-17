#!/usr/bin/env python3
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
from collections import OrderedDict
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import retry utilities
try:
    from utils.retry_utils import retry_file_operation, RetryableError, NonRetryableError
except ImportError:
    # Fallback if retry_utils is not available
    def retry_file_operation():
        def decorator(func):
            return func
        return decorator
    


# Simple PSI-based drift detection - no external dependencies needed

def _setup_plotting():
    """Helper function to import and setup plotting libraries"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        return plt, sns
    except ImportError as e:
        logger.error(f"Plotting libraries not available: {e}")
        return None, None

def _save_plot(plt, output_dir: str, filename: str, title: str = None):
    """Helper function to save plots with consistent settings"""
    os.makedirs(output_dir, exist_ok=True)
    if title:
        plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_path

def _log_error_and_return(logger_instance, operation: str, error: Exception, return_value=None):
    """Helper function for consistent error logging"""
    logger_instance.error(f"Error in {operation}: {error}")
    return {'status': 'error', 'message': str(error)} if return_value is None else return_value

class SimpleModelMonitor:
    """Simplified model monitoring with PSI drift detection for numeric features"""
    
    def __init__(self, reference_data_path: str = "datamart/gold/training/feature_store.parquet", 
                 psi_threshold: float = 0.1, bins: int = 10):
        self.logger = self._setup_logging()
        self.reference_data = self._load_reference_data(reference_data_path)
        self.psi_threshold = psi_threshold
        self.bins = bins
    
    def _setup_logging(self) -> logging.Logger:
        """Setup basic logging"""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)
    
    def _load_data_file(self, path: str, data_type: str = "data") -> Optional[pd.DataFrame]:
        """Robust file loader"""
        try:
            if not os.path.exists(path):
                error_msg = f"{data_type} file not found: {path}"
                self.logger.error(error_msg) if data_type == "Current data" else self.logger.warning(error_msg)
                raise NonRetryableError(error_msg)
            
            # Check file size to avoid loading empty files
            file_size = os.path.getsize(path)
            if file_size == 0:
                error_msg = f"{data_type} file is empty: {path}"
                self.logger.warning(error_msg)
                raise NonRetryableError(error_msg)
            
            # Load based on file extension
            if path.endswith('.parquet'):
                data = pd.read_parquet(path)
            elif path.endswith('.jsonl'):
                data = pd.read_json(path, lines=True)
            elif path.endswith('.csv'):
                # Try to detect if it's actually JSON format
                with open(path, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('{') or first_line.startswith('['):
                        self.logger.info(f"Detected JSON format in {path}")
                        data = self._load_multiline_json(path)
                    else:
                        data = pd.read_csv(path)
            else:
                # Try to auto-detect format
                with open(path, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('{') or first_line.startswith('['):
                        self.logger.info(f"Auto-detected JSON format in {path}")
                        data = self._load_multiline_json(path)
                    else:
                        data = pd.read_csv(path)
            
            if data.empty:
                error_msg = f"{data_type} is empty from {path}"
                self.logger.warning(error_msg)
                raise NonRetryableError(error_msg)
            
            self.logger.info(f"{data_type} loaded: {len(data)} rows, {len(data.columns)} columns")
            return data
            
        except (NonRetryableError, ValueError, TypeError) as e:
            self.logger.error(f"Non-retryable error loading {data_type.lower()}: {e}")
            return None
        except Exception as e:
            error_msg = f"Error loading {data_type.lower()} from {path}: {str(e)}"
            self.logger.warning(f"Retryable error: {error_msg}")
            raise RetryableError(error_msg)
    
    @retry_file_operation()
    def _load_reference_data(self, path: str) -> Optional[pd.DataFrame]:
        """Load reference data for comparison with retry logic"""
        return self._load_data_file(path, "Reference data")
    
    def _load_multiline_json(self, file_path: str) -> pd.DataFrame:
        """Load JSON data where each record spans multiple lines"""
        import json
        
        records = []
        
        with open(file_path, 'r') as f:
            content = f.read().strip()
            
            # Handle case where the file is a JSON array
            if content.startswith('[') and content.endswith(']'):
                try:
                    data = json.loads(content)
                    return pd.DataFrame(data)
                except json.JSONDecodeError:
                    pass
            
            # Handle comma-separated JSON objects
            # Split by },{ pattern and reconstruct individual objects
            if '},\n  {' in content or '},\n{' in content:
                # Split on the pattern that separates objects
                parts = content.replace('},\n  {', '}|||{').replace('},\n{', '}|||{').split('|||')
                
                for i, part in enumerate(parts):
                    part = part.strip()
                    if not part:
                        continue
                    
                    # Ensure each part is a complete JSON object
                    if not part.startswith('{'):
                        part = '{' + part
                    if not part.endswith('}'):
                        part = part + '}'
                    
                    try:
                        record = json.loads(part)
                        records.append(record)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse JSON object {i}: {e}")
                        continue
            else:
                # Handle line-by-line JSON objects
                current_record = ""
                brace_count = 0
                
                for line in content.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    
                    current_record += line
                    
                    # Count braces to detect complete JSON objects
                    brace_count += line.count('{') - line.count('}')
                    
                    # If braces are balanced, we have a complete JSON object
                    if brace_count == 0 and current_record:
                        try:
                            record = json.loads(current_record)
                            records.append(record)
                            current_record = ""
                        except json.JSONDecodeError:
                            # Continue accumulating if JSON is not complete
                            pass
        
        if not records:
            raise ValueError(f"No valid JSON records found in {file_path}")
        
        return pd.DataFrame(records)
    
    def calculate_psi(self, reference_data: pd.Series, current_data: pd.Series, bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI) for numeric features"""
        try:
            # Handle missing values
            ref_clean = reference_data.dropna()
            cur_clean = current_data.dropna()
            
            if len(ref_clean) == 0 or len(cur_clean) == 0:
                self.logger.warning("Empty data after removing NaN values")
                return float('inf')
            
            # Create bins based on reference data
            _, bin_edges = pd.cut(ref_clean, bins=bins, retbins=True, duplicates='drop')
            
            # If we have fewer unique values than bins, adjust
            if len(bin_edges) <= 2:
                unique_vals = sorted(ref_clean.unique())
                if len(unique_vals) == 1:
                    # All values are the same
                    return 0.0 if cur_clean.iloc[0] == unique_vals[0] else float('inf')
                bin_edges = unique_vals
            
            # Calculate distributions
            ref_dist, _ = pd.cut(ref_clean, bins=bin_edges, include_lowest=True).value_counts(normalize=True, sort=False).align(
                pd.cut(cur_clean, bins=bin_edges, include_lowest=True).value_counts(normalize=True, sort=False), fill_value=0)
            cur_dist, _ = pd.cut(cur_clean, bins=bin_edges, include_lowest=True).value_counts(normalize=True, sort=False).align(
                pd.cut(ref_clean, bins=bin_edges, include_lowest=True).value_counts(normalize=True, sort=False), fill_value=0)
            
            # to avoid log(0)
            epsilon = 1e-10
            ref_dist = ref_dist + epsilon
            cur_dist = cur_dist + epsilon
            
            # Calculate PSI
            psi = np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist))
            
            return float(psi)
            
        except Exception as e:
            self.logger.error(f"Error calculating PSI: {e}")
            return float('inf')
    

    
    def detect_drift_psi(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect drift using PSI for numeric features only"""
        if self.reference_data is None:
            self.logger.warning("No reference data available for PSI drift detection")
            return {"error": "No reference data available"}
        
        results = {
            "psi_results": {},
            "drift_detected": False,
            "summary": {}
        }
        
        # Get common columns
        common_columns = set(self.reference_data.columns) & set(current_data.columns)
        
        if not common_columns:
            self.logger.warning("No common columns found between reference and current data")
            return {"error": "No common columns found"}
        
        # Filter only numeric columns
        numeric_columns = []
        for col in common_columns:
            if pd.api.types.is_numeric_dtype(self.reference_data[col]):
                numeric_columns.append(col)
        
        if not numeric_columns:
            self.logger.warning("No numeric columns found for PSI calculation")
            return {"error": "No numeric columns found"}
        
        # Calculate PSI for each numeric feature
        for col in numeric_columns:
            psi_value = self.calculate_psi(
                self.reference_data[col], 
                current_data[col], 
                bins=self.bins
            )
            
            results["psi_results"][col] = {
                "psi": psi_value,
                "drift_detected": psi_value > self.psi_threshold,
                "threshold": self.psi_threshold
            }
            
            if psi_value > self.psi_threshold:
                results["drift_detected"] = True
        
        # Create summary
        drifted_features = sum(1 for col_results in results["psi_results"].values() if col_results["drift_detected"])
        
        results["summary"] = {
            "total_features_analyzed": len(numeric_columns),
            "numeric_features": len(numeric_columns),
            "features_with_drift": drifted_features,
            "drift_percentage": (drifted_features / len(numeric_columns) * 100) if len(numeric_columns) > 0 else 0
        }
        
        return results
    
    def check_data_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Check for data drift using Evidently"""
        if self.reference_data is None:
            self.logger.warning("No reference data available for drift detection")
            return self._manual_drift_detection(current_data)
        
        try:
            try:
                from evidently.report import Report
                from evidently.metric_preset import DataDriftPreset
                
                report = Report(metrics=[DataDriftPreset(),])
                report.run(reference_data=self.reference_data, current_data=current_data)
                
                # Extract results
                results = report.as_dict()
                
                # Parse drift results
                drift_detected = False
                drift_score = 0.0
                
                if 'metrics' in results:
                    for metric in results['metrics']:
                        if 'result' in metric and 'drift_detected' in metric['result']:
                            drift_detected = metric['result']['drift_detected']
                            drift_score = metric['result'].get('drift_score', 0.0)
                            break
            except ImportError:
                self.logger.warning("Evidently not installed, falling back to manual drift detection")
                return self._manual_drift_detection(current_data)
            
            return {
                "status": "DRIFT_DETECTED" if drift_detected else "OK",
                "drift_detected": drift_detected,
                "drift_score": drift_score,
                "method": "evidently"
            }
            
        except Exception as e:
            self.logger.error(f"Error in Evidently drift detection: {e}")
            return self._manual_drift_detection(current_data)
    
    def check_data_quality(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Check basic data quality metrics"""
        try:
            total_rows = len(current_data)
            missing_values = current_data.isnull().sum().sum()
            missing_percentage = (missing_values / (total_rows * len(current_data.columns))) * 100
            
            # Define quality thresholds
            quality_status = "OK"
            if missing_percentage > 20:
                quality_status = "ALERT"
            elif missing_percentage > 10:
                quality_status = "WARNING"
            
            return {
                "status": quality_status,
                "total_rows": total_rows,
                "missing_values": int(missing_values),
                "missing_percentage": round(missing_percentage, 2),
                "columns": len(current_data.columns)
            }
            
        except Exception as e:
            self.logger.error(f"Error in data quality check: {e}")
            return {"status": "ERROR", "error": str(e)}
    
    def generate_report(self, current_data: pd.DataFrame, output_dir: str = "datamart/gold/monitoring_report") -> str:
        """Generate HTML report using Evidently"""
        if self.reference_data is None:
            self.logger.warning("No reference data available for HTML report")
            return None
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create combined report
            report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
            report.run(reference_data=self.reference_data, current_data=current_data)
            
            # Save report with compatibility for different API versions
            report_path = os.path.join(output_dir, f"monitoring_report_{timestamp}.html")
            try:
                report.save_html(report_path)
            except AttributeError:
                # Try alternative methods for newer versions
                try:
                    report.save(report_path)
                except AttributeError:
                    # Generate basic HTML manually
                    self.logger.warning("Using fallback HTML generation")
                    html_content = self._generate_basic_html_report(current_data)
                    with open(report_path, 'w') as f:
                        f.write(html_content)
            
            self.logger.info(f"HTML report saved: {report_path}")
            return report_path
        
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {e}")
            return None
    
    def monitor(self, current_data_path: str, output_dir: str = "datamart/gold/monitoring_report") -> Dict[str, Any]:
        """Run PSI-based drift monitoring for numeric features"""
        try:
            # Load current data
            current_data = self._load_current_data(current_data_path)
            if current_data is None:
                return {
                    'timestamp': datetime.now().isoformat(),
                    'error': f'Failed to load current data from {current_data_path}',
                    'status': 'ERROR'
                }
            
            self.logger.info(f"Monitoring data: {len(current_data)} rows")
            
            # Check data quality
            quality_results = self.check_data_quality(current_data)
            
            # Check PSI drift detection
            psi_results = self.detect_drift_psi(current_data)
            
            # Create summary
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            monitoring_dir = os.path.join(output_dir, f"monitoring_{timestamp}")
            os.makedirs(monitoring_dir, exist_ok=True)
            
            # Determine overall status
            alerts = []
            if quality_results.get('status') == 'ALERT':
                alerts.append(f"Data quality issue: {quality_results.get('missing_percentage', 0):.1f}% missing values")
            if psi_results.get('drift_detected', False):
                summary_info = psi_results.get('summary', {})
                alerts.append(f"PSI drift detected: {summary_info.get('features_with_drift', 0)}/{summary_info.get('total_features_analyzed', 0)} features ({summary_info.get('drift_percentage', 0):.1f}%)")
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'current_data_path': current_data_path,
                'current_data_shape': current_data.shape,
                'reference_data_shape': self.reference_data.shape if self.reference_data is not None else None,
                'data_quality': quality_results,
                'psi_drift_results': psi_results,
                'alerts': alerts,
                'status': 'ALERT' if alerts else 'OK',
                'output_directory': monitoring_dir
            }
            
            # Save summary
            summary_path = os.path.join(monitoring_dir, 'monitoring_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Log results
            if alerts:
                self.logger.warning(f"Monitoring completed with {len(alerts)} alerts")
                for alert in alerts:
                    self.logger.warning(f"  - {alert}")
            else:
                self.logger.info("Monitoring completed successfully - no issues detected")
            
            # Log PSI drift detection results
            if psi_results.get("drift_detected", False):
                self.logger.warning("PSI drift detected!")
                if "summary" in psi_results:
                    summary_info = psi_results["summary"]
                    self.logger.warning(f"Features with drift: {summary_info.get('features_with_drift', 0)}/{summary_info.get('total_features_analyzed', 0)} ({summary_info.get('drift_percentage', 0):.1f}%)")
            else:
                self.logger.info("No PSI drift detected")
            
            self.logger.info(f"Summary saved to: {summary_path}")
            return results
        
        except Exception as e:
            self.logger.error(f"Error in monitoring pipeline: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'status': 'ERROR'
            }
    
    def _manual_drift_detection(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Manual drift detection using basic statistical comparison"""
        try:
            numeric_cols = self.reference_data.select_dtypes(include=[np.number]).columns
            drift_detected = False
            drift_count = 0
            
            for col in numeric_cols:
                if col in current_data.columns:
                    # Simple statistical test using mean and std comparison
                    ref_mean = self.reference_data[col].mean()
                    ref_std = self.reference_data[col].std()
                    curr_mean = current_data[col].mean()
                    
                    # Check if current mean is outside 2 standard deviations
                    if abs(curr_mean - ref_mean) > 2 * ref_std:
                        drift_count += 1
            
            drift_share = drift_count / len(numeric_cols) if len(numeric_cols) > 0 else 0
            drift_detected = drift_share > 0.2  # 20% threshold
            
            return {
                'drift_detected': drift_detected,
                'drift_share': drift_share,
                'status': 'ALERT' if drift_detected else 'OK',
                'method': 'manual_statistical'
            }
        except Exception as e:
            return {'error': f'Manual drift detection failed: {e}'}
    
    def _generate_basic_html_report(self, current_data: pd.DataFrame) -> str:
        """Generate basic HTML report when Evidently methods are not available"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Monitoring Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; }}
                .alert {{ background-color: #ffebee; }}
                .ok {{ background-color: #e8f5e8; }}
            </style>
        </head>
        <body>
            <h1>Model Monitoring Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="metric ok">
                <h3>Data Summary</h3>
                <p>Current data rows: {len(current_data)}</p>
                <p>Reference data rows: {len(self.reference_data) if self.reference_data is not None else 'N/A'}</p>
                <p>Columns: {len(current_data.columns)}</p>
            </div>
            
            <div class="metric ok">
                <h3>Data Quality</h3>
                <p>Missing values: {current_data.isnull().sum().sum()}</p>
                <p>Missing percentage: {(current_data.isnull().sum().sum() / (len(current_data) * len(current_data.columns)) * 100):.2f}%</p>
            </div>
        </body>
        </html>
        """
        return html
    
    @retry_file_operation()
    def _load_current_data(self, path: str) -> Optional[pd.DataFrame]:
        """Load current data for monitoring with retry logic"""
        return self._load_data_file(path, "Current data")
    
    def plot_model_accuracy_comparison(self, test_data_path: str, oot1_data_path: str, oot2_data_path: str, 
                                     predictions_dir: str, output_dir: str) -> Dict[str, Any]:
        """Plot model accuracy comparison across test, OOT1, and OOT2 datasets"""
        try:
            plt, _ = _setup_plotting()
            if plt is None:
                return {'status': 'error', 'message': 'Plotting libraries not available'}
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            datasets = OrderedDict([
                ('Training', {'data_path': test_data_path, 'pred_file': 'predictions_training.csv'}),
                ('OOT1', {'data_path': oot1_data_path, 'pred_file': 'predictions_oot1.csv'}),
                ('OOT2', {'data_path': oot2_data_path, 'pred_file': 'predictions_oot2.csv'})
            ])
            
            metrics_results = {}
            
            for dataset_name, paths in datasets.items():
                try:
                    # Check if data file exists first
                    if not os.path.exists(paths['data_path']):
                        self.logger.warning(f"Data file not found for {dataset_name}: {paths['data_path']}")
                        continue
                        
                    # Load actual labels using unified loading function
                    actual_data = self._load_data_file(paths['data_path'], f"{dataset_name} actual data")
                    if actual_data is None:
                        continue
                    
                    # Load predictions using unified loading function
                    pred_path = os.path.join(predictions_dir, paths['pred_file'])
                    if os.path.exists(pred_path):
                        predictions = self._load_data_file(pred_path, f"{dataset_name} predictions")
                        if predictions is None:
                            continue
                        
                        # Merge on Customer_ID
                        merged_data = actual_data.merge(predictions, on='Customer_ID', how='inner')
                        
                        if len(merged_data) > 0 and 'label' in merged_data.columns and 'prediction' in merged_data.columns:
                            y_true = merged_data['label']
                            y_pred = merged_data['prediction']
                            y_prob = merged_data.get('probability', y_pred)  # Use probability if available
                            
                            # Calculate metrics
                            accuracy = accuracy_score(y_true, y_pred)
                            precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
                            recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
                            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
                            
                            try:
                                # Ensure we have valid probability scores for AUC calculation
                                if len(np.unique(y_true)) < 2:
                                    # Cannot compute AUC with only one class
                                    self.logger.warning(f"Cannot compute AUC for {dataset_name}: only one class present")
                                    auc = 0.0
                                elif np.isnan(y_prob).any() or np.isinf(y_prob).any():
                                    # Handle invalid probability values
                                    self.logger.warning(f"Invalid probability values detected for {dataset_name}, using predictions instead")
                                    auc = roc_auc_score(y_true, y_pred)
                                else:
                                    auc = roc_auc_score(y_true, y_prob)
                            except ValueError as e:
                                self.logger.warning(f"AUC calculation failed for {dataset_name}: {e}. Using binary predictions.")
                                try:
                                    auc = roc_auc_score(y_true, y_pred)
                                except ValueError as e2:
                                    self.logger.error(f"AUC calculation completely failed for {dataset_name}: {e2}")
                                    auc = 0.0
                            except Exception as e:
                                self.logger.error(f"Unexpected error in AUC calculation for {dataset_name}: {e}")
                                auc = 0.0
                            
                            metrics_results[dataset_name] = {
                                'accuracy': accuracy,
                                'precision': precision,
                                'recall': recall,
                                'f1_score': f1,
                                'auc': auc,
                                'sample_size': len(merged_data)
                            }
                        else:
                            self.logger.warning(f"No valid data found for {dataset_name}")
                    else:
                        self.logger.warning(f"Predictions file not found: {pred_path}")
                        
                except Exception as e:
                    self.logger.error(f"Error processing {dataset_name}: {e}")
            
            if metrics_results:
                # Create a single model accuracy comparison plot with all metrics
                fig, ax = plt.subplots(figsize=(15, 8))
                
                # Ensure consistent dataset order
                dataset_order = list(datasets.keys())
                # Filter to only include datasets that have metrics
                datasets_list = [ds for ds in dataset_order if ds in metrics_results]
                
                metrics_list = ['accuracy', 'precision', 'recall', 'f1_score']
                metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                markers = ['o', 's', '^', 'd']
                
                # Create a DataFrame for line plots with proper ordering
                metrics_df = pd.DataFrame(index=datasets_list)
                for metric in metrics_list:
                    metrics_df[metric] = [metrics_results[ds][metric] for ds in datasets_list]
                
                # Plot all metrics on a single chart with different colors and markers
                for i, (metric, name, color, marker) in enumerate(zip(metrics_list, metric_names, colors, markers)):
                    # Plot the line
                    metrics_df[metric].plot(kind='line', ax=ax, marker=marker, linewidth=2, 
                                           color=color, markersize=8, label=name)
                
                # Add labels and styling
                ax.set_title('Model Performance Comparison: Training vs OOT1 vs OOT2', fontsize=14)
                ax.set_xlabel('Dataset', fontsize=12)
                ax.set_ylabel('Score', fontsize=12)
                ax.set_ylim(0, 1)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend(fontsize=10)
                
                # Ensure x-axis labels are in the correct order
                ax.set_xticks(range(len(datasets_list)))
                ax.set_xticklabels(datasets_list)
                
                # Add value labels at each point
                for i, metric in enumerate(metrics_list):
                    for j, value in enumerate(metrics_df[metric]):
                        # Adjust vertical position to avoid overlapping labels
                        offset = 0.02 + (i * 0.015)
                        ax.text(j, value + offset, f'{value:.3f}', ha='center', 
                                va='bottom', fontsize=9, color=colors[i])
                
                # Set x-ticks to dataset names
                ax.set_xticks(range(len(datasets_list)))
                ax.set_xticklabels(datasets_list)
                
                accuracy_plot_path = _save_plot(plt, output_dir, 'model_accuracy_comparison.png', 
                                               'Model Performance Comparison: Training vs OOT1 vs OOT2')
                
                self.logger.info(f"Accuracy comparison plot saved: {accuracy_plot_path}")
                return {'status': 'success', 'metrics': metrics_results, 'plot_path': accuracy_plot_path}
            else:
                return {'status': 'error', 'message': 'No valid metrics calculated'}
                
        except Exception as e:
            self.logger.error(f"Error in accuracy comparison: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def plot_distribution_comparison(self, test_data_path: str, oot1_data_path: str, oot2_data_path: str,
                                   predictions_dir: str, output_dir: str) -> Dict[str, Any]:
        """Plot y and y_predict distribution comparison across training, test, OOT1, and OOT2"""
        try:
            plt, sns = _setup_plotting()
            if plt is None:
                return {'status': 'error', 'message': 'Plotting libraries not available'}
            
            datasets = OrderedDict([
                ('Training', {'data_path': test_data_path, 'pred_file': 'predictions_training.csv'}),
                ('OOT1', {'data_path': oot1_data_path, 'pred_file': 'predictions_oot1.csv'}),
                ('OOT2', {'data_path': oot2_data_path, 'pred_file': 'predictions_oot2.csv'})
            ])
            
            all_data = []
            
            for dataset_name, paths in datasets.items():
                try:
                    # Check if data file exists first
                    if not os.path.exists(paths['data_path']):
                        self.logger.warning(f"Data file not found for {dataset_name}: {paths['data_path']}")
                        continue
                        
                    # Load actual labels using unified loading function
                    actual_data = self._load_data_file(paths['data_path'], f"{dataset_name} actual data")
                    if actual_data is None:
                        continue
                    
                    # Load predictions using unified loading function
                    pred_path = os.path.join(predictions_dir, paths['pred_file'])
                    if os.path.exists(pred_path):
                        predictions = self._load_data_file(pred_path, f"{dataset_name} predictions")
                        if predictions is None:
                            continue
                        
                        # Merge on Customer_ID
                        merged_data = actual_data.merge(predictions, on='Customer_ID', how='inner')
                        
                        if len(merged_data) > 0 and 'label' in merged_data.columns and 'prediction' in merged_data.columns:
                            merged_data['dataset'] = dataset_name
                            all_data.append(merged_data[['label', 'prediction', 'dataset']])
                        
                except Exception as e:
                    self.logger.error(f"Error processing {dataset_name} for distribution: {e}")
            
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                
                # Create a single distribution plot with all metrics
                fig, ax = plt.subplots(figsize=(15, 8))
                
                # Prepare data for plotting
                label_counts = combined_data.groupby(['dataset', 'label']).size().unstack(fill_value=0)
                pred_counts = combined_data.groupby(['dataset', 'prediction']).size().unstack(fill_value=0)
                
                # Calculate rates for plotting and ensure consistent dataset order
                dataset_order = list(datasets.keys())  # Use the same order as defined in datasets OrderedDict
                
                # Create DataFrame with proper ordering
                comparison_df = pd.DataFrame(index=dataset_order)
                
                # Fill with data
                for dataset in dataset_order:
                    dataset_data = combined_data[combined_data['dataset'] == dataset]
                    if len(dataset_data) > 0:
                        comparison_df.loc[dataset, 'Actual Label 1'] = dataset_data['label'].mean()
                        comparison_df.loc[dataset, 'Predicted Label 1'] = dataset_data['prediction'].mean()
                
                comparison_df['Difference'] = comparison_df['Predicted Label 1'] - comparison_df['Actual Label 1']
                
                # Plot all metrics on a single chart
                comparison_df['Actual Label 1'].plot(kind='line', ax=ax, color='#1f77b4', marker='o', linewidth=2, label='Actual Positive Rate')
                comparison_df['Predicted Label 1'].plot(kind='line', ax=ax, color='#d62728', marker='s', linewidth=2, label='Predicted Positive Rate')
                comparison_df['Difference'].plot(kind='line', ax=ax, color='#ff7f0e', marker='^', linewidth=2, label='Difference (Pred - Actual)')
                
                # Add horizontal line at y=0 for reference
                ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                
                # Add labels and styling
                ax.set_title('Label and Prediction Distributions: Training vs OOT1 vs OOT2', fontsize=14)
                ax.set_xlabel('Dataset', fontsize=12)
                ax.set_ylabel('Rate', fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend(fontsize=10)
                
                # Ensure x-axis labels are in the correct order
                ax.set_xticks(range(len(dataset_order)))
                ax.set_xticklabels(dataset_order)
                
                # Add value labels at each point
                for col in ['Actual Label 1', 'Predicted Label 1', 'Difference']:
                    for i, value in enumerate(comparison_df[col]):
                        y_offset = 0.02 if col != 'Difference' else -0.02
                        ax.text(i, value + y_offset, f'{value:.3f}', ha='center', va='bottom' if y_offset > 0 else 'top', fontsize=9)
                
                dist_plot_path = _save_plot(plt, output_dir, 'label_distribution_comparison.png',
                                           'Label and Prediction Distributions: Training vs OOT1 vs OOT2')
                
                # Calculate distribution statistics
                dist_stats = {}
                for dataset in combined_data['dataset'].unique():
                    dataset_data = combined_data[combined_data['dataset'] == dataset]
                    dist_stats[dataset] = {
                        'actual_positive_rate': dataset_data['label'].mean(),
                        'predicted_positive_rate': dataset_data['prediction'].mean(),
                        'sample_size': len(dataset_data)
                    }
                
                self.logger.info(f"Distribution comparison plot saved: {dist_plot_path}")
                return {
                    'status': 'success', 
                    'distribution_stats': dist_stats, 
                    'plot_path': dist_plot_path
                }
            else:
                return {'status': 'error', 'message': 'No valid data for distribution comparison'}
                
        except Exception as e:
            self.logger.error(f"Error in distribution comparison: {e}")
            return {'status': 'error', 'message': str(e)}

def main():
    """Main function for command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PSI-based drift detection for numeric features')
    parser.add_argument('--current-data', required=True, help='Path to current data file')
    parser.add_argument('--reference-data', default='datamart/gold/training/feature_store.parquet', 
                       help='Path to reference data file')
    parser.add_argument('--output-dir', default='datamart/gold/monitoring_report', help='Output directory for reports')
    parser.add_argument('--psi-threshold', type=float, default=0.1, help='PSI threshold for drift detection')
    parser.add_argument('--bins', type=int, default=10, help='Number of bins for PSI calculation')
    
    # New arguments for OOT analysis
    parser.add_argument('--test-data', help='Path to test dataset for accuracy comparison')
    parser.add_argument('--oot1-data', help='Path to OOT1 dataset')
    parser.add_argument('--oot2-data', help='Path to OOT2 dataset')
    parser.add_argument('--predictions-dir', default='datamart/predictions', help='Directory containing prediction files')
    parser.add_argument('--enable-oot-analysis', action='store_true', help='Enable OOT accuracy and distribution analysis')
    
    args = parser.parse_args()
    
    # Initialize monitor with PSI parameters
    monitor = SimpleModelMonitor(
        reference_data_path=args.reference_data,
        psi_threshold=args.psi_threshold,
        bins=args.bins
    )
    
    # Run monitoring
    results = monitor.monitor(args.current_data, args.output_dir)
    
    # Run OOT analysis if enabled
    if args.enable_oot_analysis and args.test_data and args.oot1_data and args.oot2_data:
        logger.info("Running OOT Analysis")
        
        # Run accuracy comparison
        accuracy_results = monitor.plot_model_accuracy_comparison(
            args.test_data, args.oot1_data, args.oot2_data, 
            args.predictions_dir, args.output_dir
        )
        
        if accuracy_results['status'] == 'success':
            logger.info("Model Accuracy Comparison")
            for dataset, metrics in accuracy_results['metrics'].items():
                logger.info(f"{dataset} Dataset:")
                logger.info(f"  Accuracy: {metrics['accuracy']:.3f}")
                logger.info(f"  Precision: {metrics['precision']:.3f}")
                logger.info(f"  Recall: {metrics['recall']:.3f}")
                logger.info(f"  F1-Score: {metrics['f1_score']:.3f}")
                logger.info(f"  AUC: {metrics['auc']:.3f}")
                logger.info(f"  Sample Size: {metrics['sample_size']}")
            logger.info(f"Accuracy plot saved: {accuracy_results['plot_path']}")
        else:
            logger.error(f"Accuracy analysis failed: {accuracy_results.get('message', 'Unknown error')}")
        
        # Run distribution comparison
        dist_results = monitor.plot_distribution_comparison(
            args.test_data, args.oot1_data, args.oot2_data,
            args.predictions_dir, args.output_dir
        )
        
        if dist_results['status'] == 'success':
            logger.info("Distribution Comparison")
            for dataset, stats in dist_results['distribution_stats'].items():
                logger.info(f"{dataset} Dataset:")
                logger.info(f"  Actual Positive Rate: {stats['actual_positive_rate']:.3f}")
                logger.info(f"  Predicted Positive Rate: {stats['predicted_positive_rate']:.3f}")
                logger.info(f"  Sample Size: {stats['sample_size']}")
            logger.info(f"Distribution plot saved: {dist_results['plot_path']}")
        else:
            logger.error(f"Distribution analysis failed: {dist_results.get('message', 'Unknown error')}")
    
    # Display results
    logger.info("Monitoring Results")
    logger.info(f"Timestamp: {results.get('timestamp', 'N/A')}")
    logger.info(f"Current data shape: {results.get('current_data_shape', 'N/A')}")
    logger.info(f"Reference data shape: {results.get('reference_data_shape', 'N/A')}")
    
    if 'error' in results:
        logger.error(f"Error: {results['error']}")
    else:
        quality_results = results.get('data_quality', {})
        psi_results = results.get('psi_drift_results', {})
        
        logger.info(f"Data Quality Status: {quality_results.get('status', 'Unknown')}")
        logger.info(f"Missing Values: {quality_results.get('missing_percentage', 0):.1f}%")
        
        # Display PSI results
        if psi_results:
            logger.info(f"PSI Drift Detection:")
            logger.info(f"  Overall drift detected: {psi_results.get('drift_detected', False)}")
            if 'summary' in psi_results:
                summary = psi_results['summary']
                logger.info(f"  Numeric features analyzed: {summary.get('total_features_analyzed', 0)}")
                logger.info(f"  Features with drift: {summary.get('features_with_drift', 0)}")
                logger.info(f"  Drift percentage: {summary.get('drift_percentage', 0):.1f}%")
            
            # Display individual PSI values
            if 'psi_results' in psi_results:
                logger.info(f"  Individual PSI values:")
                for feature, psi_info in psi_results['psi_results'].items():
                    status = "DRIFT" if psi_info['drift_detected'] else "OK"
                    logger.info(f"    {feature}: {psi_info['psi']:.4f} ({status})")
        
        logger.info(f"Overall Status: {results.get('status', 'Unknown')}")
        if results.get('alerts'):
            logger.warning(f"Alerts: {len(results['alerts'])}")
            for alert in results['alerts']:
                logger.warning(f"  - {alert}")
    
    return results

if __name__ == "__main__":
    main()