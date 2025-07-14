"""
Configuration file for the Student Performance Prediction project.
Contains all hyperparameters, paths, and settings.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class Config:
    """Main configuration class for the project."""
    
    # Data paths
    DATA_PATH: str = "StudentPerformance.csv"
    OUTPUT_DIR: str = "outputs"
    MODELS_DIR: str = "models"
    FIGURES_DIR: str = "figures"
    
    # Data preprocessing
    TEST_SIZE: float = 0.2
    VALIDATION_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    
    # Feature engineering
    NUMERICAL_FEATURES: List[str] = None
    CATEGORICAL_FEATURES: List[str] = None
    
    TARGET_COLUMN: str = 'Class'
    TARGET_MAPPING: Dict[str, int] = None
    TARGET_LABELS: List[str] = None
    
    # Model configurations
    MODELS_CONFIG: Dict[str, Dict[str, Any]] = None
    
    # Cross-validation settings
    CV_FOLDS: int = 5
    SCORING_METRICS: List[str] = None
    
    # Visualization settings
    FIGURE_SIZE: tuple = (12, 8)
    DPI: int = 300
    STYLE: str = 'seaborn-v0_8'
    
    # Feature selection
    FEATURE_SELECTION_METHODS: List[str] = None
    MAX_FEATURES: int = 15
    
    def __post_init__(self):
        """Initialize mutable defaults and create necessary directories."""
        # Initialize feature lists
        if self.NUMERICAL_FEATURES is None:
            self.NUMERICAL_FEATURES = [
                'raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion'
            ]
        
        if self.CATEGORICAL_FEATURES is None:
            self.CATEGORICAL_FEATURES = [
                'gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID', 
                'SectionID', 'Topic', 'Semester', 'Relation', 'ParentAnsweringSurvey',
                'ParentschoolSatisfaction', 'StudentAbsenceDays'
            ]
        
        # Initialize target mapping and labels
        if self.TARGET_MAPPING is None:
            self.TARGET_MAPPING = {'L': 0, 'M': 1, 'H': 2}
        
        if self.TARGET_LABELS is None:
            self.TARGET_LABELS = ['Low', 'Medium', 'High']
        
        # Initialize model configurations
        if self.MODELS_CONFIG is None:
            self.MODELS_CONFIG = {
                'LogisticRegression': {
                    'param_grid': {
                        'C': [0.1, 1, 10, 100],
                        'penalty': ['l1', 'l2'],
                        'solver': ['liblinear', 'saga']
                    }
                },
                'RandomForest': {
                    'param_grid': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                },
                'GradientBoosting': {
                    'param_grid': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7],
                        'subsample': [0.8, 0.9, 1.0]
                    }
                },
                'SVM': {
                    'param_grid': {
                        'C': [0.1, 1, 10, 100],
                        'kernel': ['rbf', 'poly', 'sigmoid'],
                        'gamma': ['scale', 'auto']
                    }
                },
                'XGBoost': {
                    'param_grid': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7],
                        'subsample': [0.8, 0.9, 1.0],
                        'colsample_bytree': [0.8, 0.9, 1.0]
                    }
                }
            }
        
        # Initialize scoring metrics
        if self.SCORING_METRICS is None:
            self.SCORING_METRICS = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        # Initialize feature selection methods
        if self.FEATURE_SELECTION_METHODS is None:
            self.FEATURE_SELECTION_METHODS = ['mutual_info', 'chi2', 'f_classif']
        
        # Create necessary directories
        for directory in [self.OUTPUT_DIR, self.MODELS_DIR, self.FIGURES_DIR]:
            os.makedirs(directory, exist_ok=True)

# Global configuration instance
config = Config()
