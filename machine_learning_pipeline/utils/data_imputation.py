#!/usr/bin/env python3
"""
Advanced Missing Data Imputation Module

This module provides sophisticated missing data handling strategies for the ML pipeline,
including feature-specific imputation methods and consistency between training and prediction.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import joblib
import logging
import os
from typing import Dict, List, Optional, Union, Tuple
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedDataImputer:
    """
    Advanced data imputation class that handles missing values with multiple strategies:
    - Numeric features: Median, Mean, KNN, or Iterative imputation
    - Categorical features: Mode, constant value, or frequency-based imputation
    - Feature-specific strategies based on data characteristics
    """
    
    def __init__(self, config_path: str = 'utils/config.yaml'):
        """
        Initialize the imputer with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.imputers = {}
        self.feature_strategies = {}
        self.feature_stats = {}
        self.is_fitted = False
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('imputation', self._get_default_config())
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default settings.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default imputation configuration."""
        return {
            'numeric_strategy': 'median',  # 'mean', 'median', 'knn', 'iterative'
            'categorical_strategy': 'most_frequent',  # 'most_frequent', 'constant'
            'knn_neighbors': 5,
            'iterative_max_iter': 10,
            'missing_threshold': 0.5,  # Drop features with >50% missing
            'feature_specific': {
                'Age': 'median',
                'Annual_Income': 'median', 
                'Monthly_Inhand_Salary': 'median',
                'Interest_Rate': 'median',
                'Num_of_Loan': 'median',
                'Num_Credit_Card': 'median',
                'Num_Bank_Accounts': 'median',
                'Outstanding_Debt': 'median',
                'Credit_Utilization_Ratio': 'median',
                'Total_EMI_per_month': 'median',
                'Monthly_Balance': 'median',
                'Num_of_Delayed_Payment': 'median',
                'Delay_from_due_date': 'median',
                'Debt_to_Income_Ratio': 'median',
                'EMI_Burden_Ratio': 'median',
                'Credit_Mix_encoded': 'most_frequent'
            }
        }
    
    def _analyze_missing_patterns(self, data: pd.DataFrame) -> Dict:
        """
        Analyze missing data patterns to inform imputation strategy.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary with missing data analysis
        """
        missing_info = {}
        
        for column in data.columns:
            missing_count = data[column].isnull().sum()
            missing_pct = (missing_count / len(data)) * 100
            
            missing_info[column] = {
                'missing_count': missing_count,
                'missing_percentage': missing_pct,
                'dtype': str(data[column].dtype),
                'unique_values': data[column].nunique(),
                'is_numeric': pd.api.types.is_numeric_dtype(data[column])
            }
            
        return missing_info
    
    def _detect_masked_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and convert masked missing values to proper NaN.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with masked values converted to NaN
        """
        data_clean = data.copy()
        
        # Common patterns for masked missing values
        masked_patterns = [
            '#F%$D@*&8',  # Masked SSN
            '_______',    # Masked occupation
            '___',
            'NULL',
            'null',
            'N/A',
            'n/a',
            '',
            ' ',
            '?',
            '-',
            'Unknown',
            'unknown'
        ]
        
        for column in data_clean.columns:
            # Handle string columns
            if data_clean[column].dtype == 'object':
                for pattern in masked_patterns:
                    data_clean[column] = data_clean[column].replace(pattern, np.nan)
                
                # Handle values with trailing underscores (like '40_')
                if column == 'Age':
                    data_clean[column] = data_clean[column].astype(str).str.replace('_', '')
                    data_clean[column] = pd.to_numeric(data_clean[column], errors='coerce')
            
            # Handle numeric columns with impossible values
            elif pd.api.types.is_numeric_dtype(data_clean[column]):
                # Replace negative values for features that shouldn't be negative
                if column in ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
                             'Num_Credit_Card', 'Num_of_Loan']:
                    data_clean.loc[data_clean[column] < 0, column] = np.nan
                
                # Replace extremely large values that are likely errors
                if column in ['Age']:
                    data_clean.loc[data_clean[column] > 120, column] = np.nan
                    data_clean.loc[data_clean[column] < 18, column] = np.nan
        
        return data_clean
    
    def fit(self, data: pd.DataFrame) -> 'AdvancedDataImputer':
        """
        Fit the imputer on training data.
        
        Args:
            data: Training DataFrame
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting advanced data imputer...")
        
        # Clean masked missing values
        data_clean = self._detect_masked_missing_values(data)
        
        # Analyze missing patterns
        missing_analysis = self._analyze_missing_patterns(data_clean)
        
        # Log missing data analysis
        logger.info("Missing Data Analysis:")
        for col, info in missing_analysis.items():
            if info['missing_count'] > 0:
                logger.info(f"  {col}: {info['missing_count']} missing ({info['missing_percentage']:.2f}%)")
        
        # Drop features with too many missing values
        features_to_drop = []
        for col, info in missing_analysis.items():
            if info['missing_percentage'] > (self.config['missing_threshold'] * 100):
                features_to_drop.append(col)
                logger.warning(f"Dropping feature {col} due to {info['missing_percentage']:.2f}% missing values")
        
        if features_to_drop:
            data_clean = data_clean.drop(columns=features_to_drop)
        
        # Fit imputers for each feature
        for column in data_clean.columns:
            if data_clean[column].isnull().sum() > 0:
                strategy = self._get_feature_strategy(column, missing_analysis[column])
                self.feature_strategies[column] = strategy
                
                if strategy in ['mean', 'median', 'most_frequent', 'constant']:
                    imputer = SimpleImputer(strategy=strategy, fill_value=0 if strategy == 'constant' else None)
                    self.imputers[column] = imputer.fit(data_clean[[column]])
                    
                elif strategy == 'knn':
                    # Use KNN imputer for this feature
                    numeric_cols = data_clean.select_dtypes(include=[np.number]).columns
                    if column in numeric_cols and len(numeric_cols) > 1:
                        knn_data = data_clean[numeric_cols]
                        imputer = KNNImputer(n_neighbors=self.config['knn_neighbors'])
                        self.imputers[column] = imputer.fit(knn_data)
                    else:
                        # Fallback to median for single numeric column
                        imputer = SimpleImputer(strategy='median')
                        self.imputers[column] = imputer.fit(data_clean[[column]])
                        
                elif strategy == 'iterative':
                    # Use iterative imputer
                    numeric_cols = data_clean.select_dtypes(include=[np.number]).columns
                    if column in numeric_cols and len(numeric_cols) > 1:
                        iterative_data = data_clean[numeric_cols]
                        imputer = IterativeImputer(max_iter=self.config['iterative_max_iter'], random_state=42)
                        self.imputers[column] = imputer.fit(iterative_data)
                    else:
                        # Fallback to median
                        imputer = SimpleImputer(strategy='median')
                        self.imputers[column] = imputer.fit(data_clean[[column]])
        
        # Store feature statistics for validation
        self.feature_stats = {
            col: {
                'mean': data_clean[col].mean() if pd.api.types.is_numeric_dtype(data_clean[col]) else None,
                'median': data_clean[col].median() if pd.api.types.is_numeric_dtype(data_clean[col]) else None,
                'mode': data_clean[col].mode().iloc[0] if len(data_clean[col].mode()) > 0 else None,
                'std': data_clean[col].std() if pd.api.types.is_numeric_dtype(data_clean[col]) else None
            }
            for col in data_clean.columns
        }
        
        self.is_fitted = True
        logger.info(f"Imputer fitted successfully for {len(self.imputers)} features")
        return self
    
    def _get_feature_strategy(self, column: str, missing_info: Dict) -> str:
        """
        Determine the best imputation strategy for a specific feature.
        
        Args:
            column: Feature name
            missing_info: Missing data information for the feature
            
        Returns:
            Imputation strategy name
        """
        # Check if feature-specific strategy is defined
        if column in self.config.get('feature_specific', {}):
            return self.config['feature_specific'][column]
        
        # Default strategy based on data type and missing percentage
        if missing_info['is_numeric']:
            if missing_info['missing_percentage'] < 10:
                return self.config['numeric_strategy']
            elif missing_info['missing_percentage'] < 30:
                return 'median'  # More robust for higher missing percentages
            else:
                return 'median'  # Conservative approach
        else:
            return self.config['categorical_strategy']
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted imputers.
        
        Args:
            data: DataFrame to transform
            
        Returns:
            DataFrame with imputed values
        """
        if not self.is_fitted:
            raise ValueError("Imputer must be fitted before transform")
        
        logger.info("Transforming data with fitted imputers...")
        
        # Clean masked missing values
        data_clean = self._detect_masked_missing_values(data.copy())
        
        # Apply imputation
        for column, imputer in self.imputers.items():
            if column in data_clean.columns and data_clean[column].isnull().sum() > 0:
                strategy = self.feature_strategies[column]
                
                if strategy in ['mean', 'median', 'most_frequent', 'constant']:
                    data_clean[column] = imputer.transform(data_clean[[column]]).ravel()
                    
                elif strategy == 'knn':
                    numeric_cols = data_clean.select_dtypes(include=[np.number]).columns
                    if column in numeric_cols:
                        # Ensure we have the same columns as during fitting
                        available_cols = [col for col in numeric_cols if col in data_clean.columns]
                        if len(available_cols) > 1:
                            knn_data = data_clean[available_cols]
                            imputed_data = imputer.transform(knn_data)
                            col_idx = list(available_cols).index(column)
                            data_clean[column] = imputed_data[:, col_idx]
                        else:
                            # Fallback to median
                            data_clean[column].fillna(self.feature_stats[column]['median'], inplace=True)
                            
                elif strategy == 'iterative':
                    numeric_cols = data_clean.select_dtypes(include=[np.number]).columns
                    if column in numeric_cols:
                        available_cols = [col for col in numeric_cols if col in data_clean.columns]
                        if len(available_cols) > 1:
                            iterative_data = data_clean[available_cols]
                            imputed_data = imputer.transform(iterative_data)
                            col_idx = list(available_cols).index(column)
                            data_clean[column] = imputed_data[:, col_idx]
                        else:
                            # Fallback to median
                            data_clean[column].fillna(self.feature_stats[column]['median'], inplace=True)
        
        # Final cleanup - fill any remaining missing values
        for column in data_clean.columns:
            if data_clean[column].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(data_clean[column]):
                    fill_value = self.feature_stats.get(column, {}).get('median', 0)
                    data_clean[column].fillna(fill_value, inplace=True)
                else:
                    fill_value = self.feature_stats.get(column, {}).get('mode', 'Unknown')
                    data_clean[column].fillna(fill_value, inplace=True)
        
        logger.info(f"Data transformation completed. Remaining missing values: {data_clean.isnull().sum().sum()}")
        return data_clean
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the imputer and transform data in one step.
        
        Args:
            data: DataFrame to fit and transform
            
        Returns:
            DataFrame with imputed values
        """
        return self.fit(data).transform(data)
    
    def save(self, filepath: str) -> None:
        """
        Save the fitted imputer to disk.
        
        Args:
            filepath: Path to save the imputer
        """
        if not self.is_fitted:
            raise ValueError("Imputer must be fitted before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        imputer_data = {
            'imputers': self.imputers,
            'feature_strategies': self.feature_strategies,
            'feature_stats': self.feature_stats,
            'config': self.config,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(imputer_data, filepath)
        logger.info(f"Imputer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'AdvancedDataImputer':
        """
        Load a fitted imputer from disk.
        
        Args:
            filepath: Path to load the imputer from
            
        Returns:
            Loaded imputer instance
        """
        imputer_data = joblib.load(filepath)
        
        imputer = cls()
        imputer.imputers = imputer_data['imputers']
        imputer.feature_strategies = imputer_data['feature_strategies']
        imputer.feature_stats = imputer_data['feature_stats']
        imputer.config = imputer_data['config']
        imputer.is_fitted = imputer_data['is_fitted']
        
        logger.info(f"Imputer loaded from {filepath}")
        return imputer
    
    def get_imputation_summary(self) -> Dict:
        """
        Get a summary of the imputation strategies used.
        
        Returns:
            Dictionary with imputation summary
        """
        if not self.is_fitted:
            return {"error": "Imputer not fitted"}
        
        summary = {
            'total_features_with_imputation': len(self.imputers),
            'strategies_used': {},
            'feature_strategies': self.feature_strategies.copy()
        }
        
        for strategy in self.feature_strategies.values():
            summary['strategies_used'][strategy] = summary['strategies_used'].get(strategy, 0) + 1
        
        return summary

def main():
    """Testing the imputation module"""
    print("Testing Advanced Data Imputation Module...")
    
    # Create sample data with missing values
    np.random.seed(42)
    data = pd.DataFrame({
        'Age': [25, np.nan, 35, 40, np.nan],
        'Income': [50000, 60000, np.nan, 80000, 55000],
        'Category': ['A', 'B', np.nan, 'A', 'C']
    })
    
    print("Original data:")
    print(data)
    print(f"Missing values: {data.isnull().sum().sum()}")
    
    # Test imputation
    imputer = AdvancedDataImputer()
    imputed_data = imputer.fit_transform(data)
    
    print("\nImputed data:")
    print(imputed_data)
    print(f"Remaining missing values: {imputed_data.isnull().sum().sum()}")
    
    print("\nImputation summary:")
    print(imputer.get_imputation_summary())

if __name__ == "__main__":
    main()