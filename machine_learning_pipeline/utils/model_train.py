import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
import joblib
import json
import yaml
from datetime import datetime
import os
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
from data_imputation import AdvancedDataImputer
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self, config_path='utils/config.yaml'):
        self.config = self.load_config(config_path)
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
        self.optimal_thresholds = {}  # Initialize optimal_thresholds dictionary
        
    def get_model_candidates(self):
        """Define all model candidates for comparison with class balancing."""
        return {
            'random_forest': RandomForestClassifier(
                n_estimators=200, max_depth=20, random_state=42, n_jobs=-1,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,  # Reduced from 200 to 100 for faster training
                max_depth=5,       # Reduced from 10 to 5 for faster training
                random_state=42,
                subsample=0.8,     # Helps with imbalanced data
                learning_rate=0.1, # Slightly higher learning rate to compensate for fewer trees
                min_samples_split=20, # Prevents overfitting and speeds up training
                verbose=0          # Disable verbose output
            ),
            'logistic_regression': LogisticRegression(
                random_state=42, max_iter=2000, C=1.0, tol=1e-4,
                class_weight='balanced'
            ),
            'svm': SVC(
                kernel='rbf', probability=True, random_state=42,
                class_weight='balanced'
            )
        }
    
    # Removed find_optimal_threshold function as requested
    
    # Plot methods removed as requested
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all model candidates with calibration and comprehensive metrics."""
        model_candidates = self.get_model_candidates()
        
        # Scale features for models that benefit from scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler for later use
        self.scaler = scaler
        
        # Initialize optimal thresholds dictionary if not already done
        if not hasattr(self, 'optimal_thresholds') or self.optimal_thresholds is None:
            self.optimal_thresholds = {}
        logger.info("Initialized optimal thresholds dictionary")
        
        for name, model in model_candidates.items():
            logger.info(f"Training {name}...")
            
            try:
                # Sample data for training to improve speed
                # The large dataset size is causing slow calibration
                # We'll use a stratified sample to maintain class distribution
                from sklearn.utils import resample
                
                # Use a smaller sample size for faster training while maintaining class distribution
                sample_size = min(10000, len(X_train))  # Cap at 10,000 samples or dataset size if smaller
                
                # Get indices for each class
                indices_0 = np.where(y_train == 0)[0]
                indices_1 = np.where(y_train == 1)[0]
                
                # Calculate proportional sample sizes
                prop_1 = len(indices_1) / len(y_train)
                n_samples_1 = int(sample_size * prop_1)
                n_samples_0 = sample_size - n_samples_1
                
                # Sample from each class
                sampled_indices_0 = resample(indices_0, n_samples=min(n_samples_0, len(indices_0)), random_state=42)
                sampled_indices_1 = resample(indices_1, n_samples=min(n_samples_1, len(indices_1)), random_state=42)
                
                # Combine indices and extract samples
                sampled_indices = np.concatenate([sampled_indices_0, sampled_indices_1])
                X_train_sampled = X_train.iloc[sampled_indices]
                y_train_sampled = y_train.iloc[sampled_indices]
                
                logger.info(f"Using {len(X_train_sampled)} samples for training (original size: {len(X_train)})")
                
                # Use scaled data for models that benefit from scaling
                if name in ['logistic_regression', 'svm']:
                    X_train_use = scaler.transform(X_train_sampled) if isinstance(X_train_sampled, pd.DataFrame) else scaler.transform(X_train_sampled)
                    X_test_use = X_test_scaled
                else:
                    X_train_use = X_train_sampled
                    X_test_use = X_test
                
                # Use the sampled y values for training
                y_train_use = y_train_sampled
                
                # Train model
                model.fit(X_train_use, y_train_use)
                
                # Apply probability calibration with optimizations
                if hasattr(model, 'predict_proba'):
                    import time
                    logger.info(f"Applying probability calibration to {name}...")
                    start_time = time.time()
                    # Reduce CV folds from 5 to 3 for faster calibration
                    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                    # Use sigmoid method which is faster than isotonic
                    calibrated_model = CalibratedClassifierCV(model, cv=cv, method='sigmoid', n_jobs=-1)
                    calibrated_model.fit(X_train_use, y_train_use)
                    end_time = time.time()
                    logger.info(f"Calibration for {name} completed in {end_time - start_time:.2f} seconds")
                    model = calibrated_model
                
                # Cross-validation score with multiple metrics - optimized for speed
                # Reduce CV folds from 5 to 3 for faster cross-validation
                logger.info(f"Starting cross-validation for {name}...")
                start_time = time.time()
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                if hasattr(model, 'predict_proba') and hasattr(model, 'classes_'):
                    # Use multiple scoring metrics in a single cross_validate call (more efficient)
                    scoring = {'auc': 'roc_auc', 'f1': 'f1', 'precision': 'precision', 'recall': 'recall'}
                    cv_results = cross_validate(model, X_train_use, y_train_use, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
                    cv_scores_auc = cv_results['test_auc']
                    cv_scores_f1 = cv_results['test_f1']
                    cv_scores_precision = cv_results['test_precision']
                    cv_scores_recall = cv_results['test_recall']
                else:
                    # Use accuracy for non-probabilistic models
                    cv_scores_auc = cross_val_score(model, X_train_use, y_train_use, cv=cv, scoring='accuracy', n_jobs=-1)
                    cv_scores_f1 = cv_scores_auc
                    cv_scores_precision = cv_scores_auc
                    cv_scores_recall = cv_scores_auc
                end_time = time.time()
                logger.info(f"Cross-validation for {name} completed in {end_time - start_time:.2f} seconds")
                
                # Get probability predictions
                if hasattr(model, 'predict_proba') and hasattr(model, 'classes_'):
                    y_pred_proba = model.predict_proba(X_test_use)[:, 1]
                    
                    # Use fixed threshold of 0.7 as requested
                    optimal_threshold = 0.7
                    self.optimal_thresholds[name] = optimal_threshold
                    logger.info(f"Using fixed threshold for {name}: {optimal_threshold:.4f}")
                    
                    # Apply optimal threshold for predictions
                    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
                else:
                    # For models without probability output
                    y_pred = model.predict(X_test_use)
                    y_pred_proba = None
                    optimal_threshold = 0.5  # Default
                    self.optimal_thresholds[name] = optimal_threshold
                    logger.info(f"Model {name} does not support probability output. Using default threshold of 0.5.")
                    
                    # For models without probability output, we still need to ensure they're in the optimal_thresholds dictionary
                    if name not in self.optimal_thresholds:
                        self.optimal_thresholds[name] = optimal_threshold
                
                # Calculate comprehensive metrics
                metrics = {}
                
                # Basic metrics
                metrics['accuracy'] = accuracy_score(y_test, y_pred)
                metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
                metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
                metrics['f1'] = f1_score(y_test, y_pred, zero_division=0)
                
                # AUC if applicable
                if y_pred_proba is not None:
                    try:
                        metrics['auc'] = roc_auc_score(y_test, y_pred_proba)
                        metrics['average_precision'] = average_precision_score(y_test, y_pred_proba)
                    except Exception as e:
                        logger.warning(f"Error calculating AUC for {name}: {e}")
                        metrics['auc'] = None
                        metrics['average_precision'] = None
                else:
                    metrics['auc'] = None
                    metrics['average_precision'] = None
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                metrics['confusion_matrix'] = cm.tolist()
                
                # Class distribution in predictions
                metrics['predicted_class_distribution'] = np.bincount(y_pred, minlength=2).tolist()
                
                # Store results
                self.models[name] = model
                self.results[name] = {
                    'cv_scores': {
                        'auc_mean': cv_scores_auc.mean(),
                        'auc_std': cv_scores_auc.std(),
                        'f1_mean': cv_scores_f1.mean(),
                        'f1_std': cv_scores_f1.std(),
                        'precision_mean': cv_scores_precision.mean(),
                        'precision_std': cv_scores_precision.std(),
                        'recall_mean': cv_scores_recall.mean(),
                        'recall_std': cv_scores_recall.std()
                    },
                    'test_metrics': metrics,
                    'optimal_threshold': optimal_threshold,
                    'model_params': model.get_params() if hasattr(model, 'get_params') else {}
                }
                
                # We'll keep the detailed structure in self.results[name] as is
                
                # Log results
                logger.info(f"{name} Results:")
                logger.info(f"  CV AUC = {cv_scores_auc.mean():.4f} (±{cv_scores_auc.std():.4f})")
                logger.info(f"  CV F1 = {cv_scores_f1.mean():.4f} (±{cv_scores_f1.std():.4f})")
                logger.info(f"  Test Accuracy = {metrics['accuracy']:.4f}")
                logger.info(f"  Test Precision = {metrics['precision']:.4f}")
                logger.info(f"  Test Recall = {metrics['recall']:.4f}")
                logger.info(f"  Test F1 = {metrics['f1']:.4f}")
                if metrics['auc'] is not None:
                    logger.info(f"  Test AUC = {metrics['auc']:.4f}")
                    logger.info(f"  Test Average Precision = {metrics['average_precision']:.4f}")
                logger.info(f"  Confusion Matrix = {metrics['confusion_matrix']}")
                logger.info(f"  Predicted Class Distribution = {metrics['predicted_class_distribution']}")
                
                # Track best model based on F1 score (better for imbalanced data)
                # F1 score is a better metric for imbalanced datasets as it balances precision and recall
                if metrics['f1'] > self.best_score:
                    self.best_score = metrics['f1']
                    self.best_model = name
                    logger.info(f"New best model: {name} with F1 score: {self.best_score:.4f}")
                    
                    # Log the optimal threshold for the best model
                    if name in self.optimal_thresholds:
                        logger.info(f"Best model optimal threshold: {self.optimal_thresholds[name]:.4f}")
                    else:
                        logger.warning(f"No optimal threshold found for best model {name}. Using default.")
                        self.optimal_thresholds[name] = 0.5
                    
            except Exception as e:
                logger.error(f"Failed to train {name}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                continue
    
    def _filter_non_serializable(self, obj):
        """Filter out non-serializable objects from dictionaries."""
        if isinstance(obj, dict):
            return {k: self._filter_non_serializable(v) for k, v in obj.items() 
                   if self._is_json_serializable(v)}
        elif isinstance(obj, list):
            return [self._filter_non_serializable(item) for item in obj 
                   if self._is_json_serializable(item)]
        else:
            return obj
    
    def _is_json_serializable(self, obj):
        """Check if an object is JSON serializable."""
        try:
            json.dumps(obj)
            return True
        except (TypeError, OverflowError):
            return False
    
    def save_model_registry(self):
        """Save all models, optimal thresholds, and comparison results."""
        registry_dir = 'datamart/registry'
        os.makedirs(registry_dir, exist_ok=True)
        
        # Save individual models with compatibility settings
        for name, model in self.models.items():
            model_path = f'{registry_dir}/{name}_model.pkl'
            joblib.dump(model, model_path, protocol=4)
        
        # Save optimal thresholds as a separate file for backward compatibility
        thresholds_path = os.path.join(registry_dir, 'optimal_thresholds.json')
        
        # Ensure we have thresholds to save
        if not self.optimal_thresholds:
            logger.warning("No optimal thresholds found to save. Using default thresholds.")
            # Create default thresholds for all models
            for name in self.models.keys():
                self.optimal_thresholds[name] = 0.5
        
        # Save the thresholds
        with open(thresholds_path, 'w') as f:
            json.dump(self.optimal_thresholds, f, indent=2)
        
        logger.info(f"Optimal thresholds saved to {thresholds_path}: {self.optimal_thresholds}")
        
        # Prepare simplified results for model_comparison.json
        simplified_results = {}
        for name, result in self.results.items():
            # Extract key metrics for simplified view
            simplified_results[name] = {
                'cv_score_mean': result['cv_scores']['f1_mean'],
                'cv_score_std': result['cv_scores']['f1_std'],
                'test_score': result['test_metrics']['f1'],
                'metric_type': 'F1',  # Explicitly set to F1 instead of AUC
                'optimal_threshold': result['optimal_threshold'],
                # Filter out non-serializable objects from model_params
                'model_params': self._filter_non_serializable(result['model_params'])
            }
            
            # Ensure optimal threshold is included in the results
            if name in self.optimal_thresholds:
                simplified_results[name]['optimal_threshold'] = self.optimal_thresholds[name]
            else:
                logger.warning(f"No optimal threshold found for model {name}")
                simplified_results[name]['optimal_threshold'] = 0.5  # Default fallback
        
        # Save comparison results
        comparison_results = {
            'timestamp': datetime.now().isoformat(),
            'best_model': self.best_model,
            'best_score': self.best_score,
            'best_metric': 'f1',  # Explicitly state that best_score is F1 score
            'optimal_thresholds': self.optimal_thresholds,  # Include optimal thresholds at top level
            'all_results': simplified_results
        }
        
        # Log the structure being saved to help with debugging
        logger.info(f"Saving model comparison with optimal thresholds: {self.optimal_thresholds}")
        logger.info(f"Best model: {self.best_model}, Best F1 score: {self.best_score:.4f}")
        if self.best_model in self.optimal_thresholds:
            logger.info(f"Best model optimal threshold: {self.optimal_thresholds[self.best_model]:.4f}")
        else:
            logger.warning(f"No optimal threshold found for best model {self.best_model}")
        
        with open(f'{registry_dir}/model_comparison.json', 'w') as f:
            json.dump(comparison_results, f, indent=2)
        logger.info(f"Model comparison results saved to {registry_dir}/model_comparison.json")
        
        # Save best model separately with compatibility settings
        if self.best_model:
            best_model_path = 'datamart/artifacts/best_model.pkl'
            os.makedirs('datamart/artifacts', exist_ok=True)
            # Use protocol 4 for better compatibility across Python versions
            joblib.dump(self.models[self.best_model], best_model_path, protocol=4)
            
            # Save best model threshold
            best_threshold_path = os.path.join('datamart/artifacts', 'best_threshold.json')
            with open(best_threshold_path, 'w') as f:
                json.dump({
                    'model_name': self.best_model,
                    'threshold': self.optimal_thresholds[self.best_model],
                    'f1_score': self.best_score
                }, f, indent=2)
            logger.info(f"Best threshold saved to {best_threshold_path}")
            
            # Save scaler if it exists
            if hasattr(self, 'scaler'):
                scaler_path = os.path.join('datamart/artifacts', 'scaler.pkl')
                joblib.dump(self.scaler, scaler_path, protocol=4)
                logger.info(f"Scaler saved to {scaler_path}")
                logger.info("Note: Data imputer is saved separately during data loading phase")
            
        logger.info(f"Best model: {self.best_model} (F1 Score: {self.best_score:.4f})")
        logger.info(f"Models saved to {registry_dir}/")
    
    # Summary report generation method removed as requested
    
    def load_config(self, config_path):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default settings.")
            return {}
    
    def load_data(self):
        """Load feature store and label store data."""
        import pandas as pd
        import os
        
        # Load training feature store data
        feature_file = 'datamart/gold/training/feature_store.parquet'
        if not os.path.exists(feature_file):
            raise FileNotFoundError(f"Training feature store file not found: {feature_file}")
        
        # Load training label store data
        label_file = 'datamart/gold/training/label_store.parquet'
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Training label store file not found: {label_file}")
        
        # Load the combined parquet files
        features_df = pd.read_parquet(feature_file)
        labels_df = pd.read_parquet(label_file)
        
        # Merge features and labels
        data = features_df.merge(labels_df, on=['Customer_ID', 'snapshot_date'], how='inner')
        
        # Prepare features and target
        feature_cols = [col for col in data.columns if col not in ['Customer_ID', 'snapshot_date', 'label', 'label_def']]
        X = data[feature_cols]
        y = data['label']
        
        # Ensure target is properly encoded for classification
        y = y.astype(int)
        
        # Handle missing values with advanced imputation
        print(f"📊 Missing values before imputation: {X.isnull().sum().sum()}")
        
        # Initialize and fit the advanced imputer
        imputer = AdvancedDataImputer(config_path='utils/config.yaml')
        X = imputer.fit_transform(X)
        
        print(f"✅ Missing values after imputation: {X.isnull().sum().sum()}")
        print("📋 Imputation summary:")
        imputation_summary = imputer.get_imputation_summary()
        for strategy, count in imputation_summary.get('strategies_used', {}).items():
            print(f"  - {strategy}: {count} features")
        
        # Save the fitted imputer for use in predictions
        imputer_path = 'datamart/artifacts/data_imputer.joblib'
        os.makedirs(os.path.dirname(imputer_path), exist_ok=True)
        imputer.save(imputer_path)
        print(f"💾 Imputer saved to {imputer_path}")
        
        logger.info(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        logger.info(f"Target data type: {y.dtype}")
        logger.info(f"Unique target values: {sorted(y.unique())}")
        logger.info(f"Data quality check: Missing values = {X.isnull().sum().sum()}, Infinite values = {np.isinf(X.select_dtypes(include=[np.number])).sum().sum()}")
        
        return X, y

def analyze_class_imbalance(y):
    """Analyze class imbalance and provide recommendations."""
    class_counts = np.bincount(y)
    total_samples = len(y)
    class_ratios = class_counts / total_samples
    
    # Calculate imbalance ratio (majority:minority)
    majority_class = np.argmax(class_counts)
    minority_class = np.argmin(class_counts)
    imbalance_ratio = class_counts[majority_class] / class_counts[minority_class]
    
    print("\n📊 Class Distribution Analysis:")
    for cls, count in enumerate(class_counts):
        print(f"  - Class {cls}: {count} samples ({class_ratios[cls]*100:.2f}%)")
    
    print(f"\n⚖️ Imbalance Ratio (majority:minority): {imbalance_ratio:.2f}:1")
    
    # Provide recommendations based on imbalance level
    print("\n🔍 Imbalance Assessment:")
    if imbalance_ratio < 3:
        print("  - Mild imbalance: Standard techniques with class weights should be sufficient.")
    elif imbalance_ratio < 10:
        print("  - Moderate imbalance: Using class weights, adjusted thresholds, and F1 score for evaluation.")
    else:
        print("  - Severe imbalance: Consider advanced techniques like SMOTE, ensemble methods, or anomaly detection approaches.")
    
    return {
        'class_counts': class_counts.tolist(),
        'class_ratios': class_ratios.tolist(),
        'imbalance_ratio': float(imbalance_ratio),
        'majority_class': int(majority_class),
        'minority_class': int(minority_class)
    }

def main():
    """Main execution function."""
    print("🚀 Starting ML Model Training Pipeline...")
    
    try:
        # Initialize model registry
        registry = ModelRegistry()
        
        # Load data
        print("📥 Loading training data...")
        X, y = registry.load_data()
        
        # Analyze class imbalance
        imbalance_info = analyze_class_imbalance(y)
        
        # Split data
        print("\n🔀 Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📈 Training set: {X_train.shape[0]} samples")
        print(f"📉 Test set: {X_test.shape[0]} samples")
        
        # Verify class distribution in splits
        train_class_dist = np.bincount(y_train)
        test_class_dist = np.bincount(y_test)
        print("\n📊 Class distribution in splits:")
        for cls in range(len(train_class_dist)):
            print(f"  - Class {cls}: {train_class_dist[cls]} (train) / {test_class_dist[cls]} (test)")
        
        # Store imbalance info for reporting
        registry.imbalance_info = imbalance_info
        
        # Train and evaluate models
        print("\n🔄 Training and evaluating models...")
        registry.train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        # Save results
        print("\n💾 Saving model registry...")
        registry.save_model_registry()
        
        print("\n✅ Model training pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in model training pipeline: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()