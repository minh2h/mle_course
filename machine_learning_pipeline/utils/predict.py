import os
import sys
import json
import pickle
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Union, Optional, Any
import yaml
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from data_imputation import AdvancedDataImputer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

class ModelPredictor:
    """
    Production-ready model prediction class that supports:
    - Loading models from registry
    - Batch and real-time predictions
    - Input validation and preprocessing
    - Prediction logging and monitoring
    - Multiple output formats
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the predictor with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.model_metadata = None
        self.feature_columns = None
        self.scaler = None
        self.imputer = None
        
        # Setup logging
        self._setup_logging()
        
        # Load model on initialization
        self._load_model()
    
    def _load_config(self, config_path: str = None) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'configs', 'config.yaml'
            )
        
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            # Start with default config and update with YAML config
            config = self._get_default_config()
            if yaml_config:
                # Deep merge the configurations
                for key, value in yaml_config.items():
                    if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                        config[key].update(value)
                    else:
                        config[key] = value
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """
        Get default configuration if config file is not available.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'prediction': {
                'model_path': 'datamart/artifacts/best_model.pkl',
                'batch_size': 1000,
                'output_format': 'json',
                'confidence_threshold': 0.1,  # Very low due to sklearn version mismatch causing poor calibration
                'log_predictions': False,
                'save_predictions_for_monitoring': True,
                'monitoring_retention_days': 30,
                'feature_store_path': 'datamart/gold/training/feature_store.parquet'
            },
            'paths': {
                'predictions_output': 'datamart/predictions/',
                'monitoring_output': 'datamart/predictions/monitoring/',
                'drift_detection_data': 'datamart/predictions/drift_detection/',
                'logs': 'datamart/logs/'
            }
        }
    
    def _setup_logging(self):
        """
        Setup logging configuration.
        """
        log_dir = self.config.get('paths', {}).get('logs', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f'predictions_{datetime.now().strftime("%Y%m%d")}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self, model_path: str = None):
        """
        Load the trained model and associated artifacts.
        
        Args:
            model_path: Path to model file
        """
        try:
            if model_path is None:
                logger.debug(f"Config: {self.config}")
                if 'prediction' not in self.config:
                    raise KeyError("'prediction' key not found in config")
                if 'model_path' not in self.config['prediction']:
                    raise KeyError("'model_path' key not found in prediction config")
                model_path = self.config['prediction']['model_path']
            
            logger.info(f"Loading model from: {model_path}")
            
            # Load model with error handling for version compatibility
            import joblib
            try:
                model_data = joblib.load(model_path)
            except Exception as e:
                self.logger.error(f"❌ Error loading best model: {e}")
                logger.error(f"Error loading best model: {e}")
                # Try loading with pickle as fallback
                try:
                    import pickle
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                    logger.info("Successfully loaded model using pickle fallback")
                except Exception as pickle_error:
                    self.logger.error(f"❌ Pickle fallback also failed: {pickle_error}")
                    raise Exception(f"Could not load model with joblib or pickle: {e}")
            
            if isinstance(model_data, dict):
                self.model = model_data['model']
                self.model_metadata = model_data.get('metadata', {})
                self.feature_columns = model_data.get('feature_columns', [])
                self.scaler = model_data.get('scaler')
            else:
                self.model = model_data
                self.model_metadata = {}
                self.feature_columns = []
                self.scaler = None
            
            # Load imputer
            imputer_path = os.path.join(os.path.dirname(model_path), 'data_imputer.joblib')
            if os.path.exists(imputer_path):
                self.imputer = AdvancedDataImputer.load(imputer_path)
                logger.info(f"Data imputer loaded from: {imputer_path}")
            else:
                logger.warning(f"Data imputer not found at: {imputer_path}. Using fallback imputation.")
                # Create a basic imputer as fallback
                self.imputer = AdvancedDataImputer(config_path='utils/config.yaml')
            
            self.logger.info(f"Model loaded successfully from {model_path}")
            self.logger.info(f"Model type: {type(self.model).__name__}")
            self.logger.info(f"Using optimal threshold: {self.config['prediction']['confidence_threshold']}")
            
            if self.model_metadata:
                self.logger.info(f"Model metadata: {self.model_metadata}")
        
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
            
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data for prediction.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        try:
            # Make a copy to avoid modifying original data
            processed_data = data.copy()
            
            # Select feature columns if available
            if self.feature_columns:
                missing_cols = set(self.feature_columns) - set(processed_data.columns)
                if missing_cols:
                    self.logger.warning(f"Missing columns: {missing_cols}")
                    # Add missing columns with default values
                    for col in missing_cols:
                        processed_data[col] = 0
                
                processed_data = processed_data[self.feature_columns]
            else:
                # If feature columns not specified, exclude known non-feature columns
                non_feature_cols = ['Customer_ID', 'snapshot_date', 'target', 'label']
                feature_cols = [col for col in processed_data.columns if col not in non_feature_cols]
                processed_data = processed_data[feature_cols]
                self.logger.info(f"Using inferred feature columns: {feature_cols}")
            
            # Handle missing values with advanced imputation
            if self.imputer is not None:
                if self.imputer.is_fitted:
                    processed_data = self.imputer.transform(processed_data)
                    self.logger.info(f"Applied fitted imputation. Remaining missing values: {processed_data.isnull().sum().sum()}")
                else:
                    # Fallback to basic imputation if imputer not fitted
                    self.logger.warning("Imputer not fitted. Using basic median/mode imputation.")
                    for col in processed_data.columns:
                        if processed_data[col].isnull().sum() > 0:
                            if pd.api.types.is_numeric_dtype(processed_data[col]):
                                processed_data[col].fillna(processed_data[col].median(), inplace=True)
                            else:
                                processed_data[col].fillna(processed_data[col].mode().iloc[0] if len(processed_data[col].mode()) > 0 else 'Unknown', inplace=True)
            else:
                # Fallback to zero-filling if no imputer available
                self.logger.warning("No imputer available. Using zero-filling for missing values.")
                processed_data = processed_data.fillna(0)
            
            # Apply scaling if scaler is available
            if self.scaler is not None:
                processed_data = pd.DataFrame(
                    self.scaler.transform(processed_data),
                    columns=processed_data.columns,
                    index=processed_data.index
                )
            
            self.logger.info(f"Preprocessed data shape: {processed_data.shape}")
            return processed_data
        
        except Exception as e:
            self.logger.error(f"Error in data preprocessing: {e}")
            raise
    
    def predict_single(self, data: Union[Dict, pd.Series, pd.DataFrame]) -> Dict:
        """
        Make prediction for a single instance using the optimal threshold.
        
        Args:
            data: Single instance data
            
        Returns:
            Prediction result dictionary
        """
        try:
            # Convert to DataFrame if needed
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, pd.Series):
                df = pd.DataFrame([data])
            else:
                df = data
            
            # Preprocess
            processed_data = self.preprocess_data(df)
            
            # Get prediction probability if available
            try:
                prediction_proba = self.model.predict_proba(processed_data)[0]
                confidence = float(prediction_proba[1])  # Probability of positive class
                
                # Use optimal threshold for prediction
                threshold = self.config['prediction']['confidence_threshold']
                prediction = 1 if confidence >= threshold else 0
                
                self.logger.info(f"Using threshold {threshold} for prediction (confidence: {confidence})")
            except Exception as e:
                self.logger.warning(f"Could not get probabilities, using default predict method: {e}")
                prediction = self.model.predict(processed_data)[0]
                prediction_proba = None
                confidence = None
            
            result = {
                'prediction': int(prediction) if isinstance(prediction, (np.integer, np.bool_)) else float(prediction),
                'confidence': confidence,
                'threshold_used': self.config['prediction']['confidence_threshold'],
                'timestamp': datetime.now().isoformat(),
                'model_version': self.model_metadata.get('version', 'unknown')
            }
            
            if prediction_proba is not None:
                result['probabilities'] = prediction_proba.tolist()
            
            # Log prediction for monitoring (no file saving for single predictions)
            self._log_prediction_for_monitoring(result)
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error in single prediction: {e}")
            raise
    
    def predict_batch(self, data: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
        """
        Make predictions for a batch of data using the optimal threshold.
        
        Args:
            data: Input DataFrame
            output_path: Optional path to save predictions
            
        Returns:
            DataFrame with predictions
        """
        try:
            self.logger.info(f"Starting batch prediction for {len(data)} samples")
            
            # Preprocess data
            processed_data = self.preprocess_data(data)
            
            # Make predictions in batches
            batch_size = self.config['prediction'].get('batch_size', 1000)
            predictions = []
            confidences = []
            threshold = self.config['prediction']['confidence_threshold']
            
            self.logger.info(f"Using threshold {threshold} for batch predictions")
            
            for i in range(0, len(processed_data), batch_size):
                batch = processed_data.iloc[i:i+batch_size]
                
                # Get probabilities if available
                try:
                    batch_probas = self.model.predict_proba(batch)
                    batch_confidences = batch_probas[:, 1]  # Probability of positive class
                    confidences.extend(batch_confidences)
                    
                    # Apply threshold to get predictions
                    batch_predictions = (batch_confidences >= threshold).astype(int)
                    predictions.extend(batch_predictions)
                except Exception as e:
                    self.logger.warning(f"Could not get probabilities, using default predict method: {e}")
                    batch_predictions = self.model.predict(batch)
                    predictions.extend(batch_predictions)
                    confidences.extend([None] * len(batch_predictions))
                
                self.logger.info(f"Processed batch {i//batch_size + 1}/{(len(processed_data)-1)//batch_size + 1}")
            
            # Create results DataFrame
            results = data.copy()
            results['prediction'] = predictions
            results['confidence'] = confidences
            results['threshold_used'] = threshold
            results['timestamp'] = datetime.now().isoformat()
            results['model_version'] = self.model_metadata.get('version', 'unknown')
            
            # Save results if output path provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Determine output format from file extension or config
                if output_path.endswith('.csv'):
                    output_format = 'csv'
                elif output_path.endswith('.json'):
                    output_format = 'json'
                elif output_path.endswith('.parquet'):
                    output_format = 'parquet'
                else:
                    output_format = self.config['prediction'].get('output_format', 'csv')
                
                if output_format == 'csv':
                    results.to_csv(output_path, index=False)
                elif output_format == 'json':
                    results.to_json(output_path, orient='records', indent=2)
                elif output_format == 'parquet':
                    results.to_parquet(output_path, index=False)
                
                self.logger.info(f"Predictions saved to {output_path} in {output_format} format")
            
            # Log batch completion
            self._save_batch_for_monitoring(results)
            
            self.logger.info(f"Batch prediction completed. Total predictions: {len(predictions)}")
            return results
        
        except Exception as e:
            self.logger.error(f"Error in batch prediction: {e}")
            raise
    
    # predict_from_feature_store method removed to simplify the code
    
    def _log_prediction_for_monitoring(self, prediction_result: Dict):
        """Log prediction for monitoring purposes (no file saving for single predictions)"""
        if self.config['prediction'].get('log_predictions', True):
            self.logger.info(f"Prediction made: {prediction_result}")
    
    def _save_batch_for_monitoring(self, results: pd.DataFrame):
        """Save batch predictions to monitoring directory for later analysis"""
        self.logger.info(f"Batch prediction completed with {len(results)} results")
        
        # Save batch predictions for monitoring if configured
        if self.config['prediction'].get('save_predictions_for_monitoring', False):
            try:
                # Create monitoring directory if it doesn't exist
                monitoring_dir = self.config['paths'].get('monitoring_output', 'datamart/predictions/monitoring/')
                os.makedirs(monitoring_dir, exist_ok=True)
                
                # Create a unique filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(monitoring_dir, f"batch_prediction_{timestamp}.json")
                
                # Save to file
                results.to_json(filename, orient='records', indent=2)
                    
                self.logger.info(f"Batch predictions saved for monitoring: {filename}")
            except Exception as e:
                self.logger.error(f"Error saving batch predictions for monitoring: {e}")
                # Don't raise the exception to avoid disrupting the prediction flow
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Model information dictionary
        """
        return {
            'model_type': type(self.model).__name__ if self.model else None,
            'metadata': self.model_metadata,
            'feature_columns': self.feature_columns,
            'has_scaler': self.scaler is not None,
            'has_imputer': self.imputer is not None,
            'config': self.config
        }

def main():
    """
    Main function for command-line usage.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Prediction Script')
    parser.add_argument('--mode', choices=['single', 'batch'], 
                       default='single', help='Prediction mode')
    parser.add_argument('--input', help='Input file path for batch prediction')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--config', help='Config file path')
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = ModelPredictor(config_path=args.config)
        
        # Model is automatically loaded during initialization
        print("🚀 Initializing ML Prediction System...")
        print("✅ Model loaded successfully")
        
        if args.mode == 'single':
            # Example single prediction
            sample_data = {
                'Annual_Income': 50000.0,
                'Monthly_Inhand_Salary': 4000.0,
                'Num_Bank_Accounts': 2,
                'Num_Credit_Card': 1,
                'Interest_Rate': 10.0,
                'Num_of_Loan': 1,
                'Outstanding_Debt': 5000.0,
                'Credit_Utilization_Ratio': 30.0,
                'Total_EMI_per_month': 1000.0,
                'Monthly_Balance': 3000.0,
                'Num_of_Delayed_Payment': 0,
                'Delay_from_due_date': 0,
                'Age': 30,
                'Credit_Mix_encoded': 2,
                'Debt_to_Income_Ratio': 10.0,
                'EMI_Burden_Ratio': 25.0
            }
            result = predictor.predict_single(sample_data)
            print(json.dumps(result, indent=2))
        
        elif args.mode == 'batch':
            if not args.input:
                raise ValueError("Input file required for batch prediction")
            
            # Read data based on file extension
            if args.input.endswith('.parquet'):
                data = pd.read_parquet(args.input)
            elif args.input.endswith('.csv'):
                data = pd.read_csv(args.input)
            else:
                raise ValueError(f"Unsupported file format. Expected .csv or .parquet, got: {args.input}")
            
            results = predictor.predict_batch(data, args.output)
            print(f"Batch prediction completed. Results shape: {results.shape}")
        
        # Print model info
        model_info = predictor.get_model_info()
        print(f"\nModel Info: {json.dumps(model_info, indent=2, default=str)}")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())