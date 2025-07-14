"""
Model deployment and prediction service.
Provides functionality to load trained models and make predictions on new data.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from typing import Dict, List, Union, Tuple
from config import config

warnings.filterwarnings('ignore')

class StudentPerformancePredictor:
    """Model deployment class for making predictions on new student data."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize the predictor.
        
        Args:
            model_name: Name of the model to load. If None, loads the best model.
        """
        self.model_name = model_name
        self.model = None
        self.preprocessing_objects = None
        self.feature_names = None
        self.loaded = False
        
    def load_model(self, model_path: str = None):
        """
        Load a trained model and preprocessing objects.
        
        Args:
            model_path: Path to the model file. If None, uses default path.
        """
        try:
            # Load preprocessing objects
            preprocessing_path = f"{config.MODELS_DIR}/preprocessing_objects.pkl"
            self.preprocessing_objects = joblib.load(preprocessing_path)
            
            # Extract preprocessing components
            self.scaler = self.preprocessing_objects['scaler']
            self.le_target = self.preprocessing_objects['le_target']
            self.feature_names = self.preprocessing_objects['feature_names']
            self.numerical_features = self.preprocessing_objects['numerical_features']
            self.categorical_features = self.preprocessing_objects['categorical_features']
            
            # Load model
            if model_path is None:
                if self.model_name is None:
                    # Try to find the best model
                    try:
                        evaluation_results = joblib.load(f"{config.MODELS_DIR}/evaluation_results.pkl")
                        self.model_name = max(evaluation_results, 
                                            key=lambda x: evaluation_results[x]['accuracy'])
                    except:
                        self.model_name = "XGBoost"  # Default fallback
                
                model_path = f"{config.MODELS_DIR}/{self.model_name}_model.pkl"
            
            self.model = joblib.load(model_path)
            self.loaded = True
            
            print(f"Model loaded successfully: {self.model_name}")
            print(f"Expected features: {len(self.feature_names)}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_input(self, input_data: Union[Dict, pd.DataFrame]) -> np.ndarray:
        """
        Preprocess input data for prediction.
        
        Args:
            input_data: Input data as dictionary or DataFrame
            
        Returns:
            Preprocessed feature array
        """
        if not self.loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Convert to DataFrame if necessary
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Add engineered features (simplified version)
        df = self._add_engineered_features(df)
        
        # Separate numerical and categorical features
        numerical_cols = [col for col in df.columns if col in self.numerical_features]
        categorical_cols = [col for col in df.columns if col in self.categorical_features]
        
        # Handle numerical features
        X_numerical = df[numerical_cols].fillna(df[numerical_cols].median())
        X_numerical_scaled = pd.DataFrame(
            self.scaler.transform(X_numerical),
            columns=numerical_cols,
            index=X_numerical.index
        )
        
        # Handle categorical features
        X_categorical = df[categorical_cols].fillna('Unknown')
        X_categorical_encoded = pd.get_dummies(X_categorical, prefix=categorical_cols, drop_first=True)
        
        # Combine features
        X_processed = pd.concat([X_numerical_scaled, X_categorical_encoded], axis=1)
        
        # Ensure all expected features are present
        for feature in self.feature_names:
            if feature not in X_processed.columns:
                X_processed[feature] = 0
        
        # Select only the features used during training
        X_processed = X_processed[self.feature_names]
        
        return X_processed.values
    
    def _add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features to the input data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df_eng = df.copy()
        
        # Engagement features
        engagement_features = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']
        available_engagement = [f for f in engagement_features if f in df_eng.columns]
        
        if available_engagement:
            df_eng['engagement_score'] = df_eng[available_engagement].sum(axis=1)
            df_eng['engagement_score_normalized'] = (df_eng['engagement_score'] - 
                                                   df_eng['engagement_score'].min()) / (
                                                   df_eng['engagement_score'].max() - 
                                                   df_eng['engagement_score'].min() + 1e-6)
        
        # Participation features
        if 'raisedhands' in df_eng.columns and 'Discussion' in df_eng.columns:
            df_eng['participation_rate'] = (df_eng['raisedhands'] + df_eng['Discussion']) / 2
            df_eng['hands_to_discussion_ratio'] = (df_eng['raisedhands'] + 1) / (df_eng['Discussion'] + 1)
        
        # Digital engagement
        if 'VisITedResources' in df_eng.columns and 'AnnouncementsView' in df_eng.columns:
            df_eng['digital_engagement'] = (df_eng['VisITedResources'] + df_eng['AnnouncementsView']) / 2
            df_eng['resources_to_announcements_ratio'] = (df_eng['VisITedResources'] + 1) / (df_eng['AnnouncementsView'] + 1)
        
        # Grade features
        if 'GradeID' in df_eng.columns:
            df_eng['grade_numeric'] = df_eng['GradeID'].str.extract('(\\d+)').astype(float)
            df_eng['is_high_grade'] = (df_eng['grade_numeric'] > 6).astype(int)
        
        # Parent involvement
        parent_involvement = 0
        if 'ParentAnsweringSurvey' in df_eng.columns:
            parent_involvement += (df_eng['ParentAnsweringSurvey'] == 'Yes').astype(int)
        if 'ParentschoolSatisfaction' in df_eng.columns:
            parent_involvement += (df_eng['ParentschoolSatisfaction'] == 'Good').astype(int)
        df_eng['parent_involvement_score'] = parent_involvement
        
        # Attendance
        if 'StudentAbsenceDays' in df_eng.columns:
            df_eng['good_attendance'] = (df_eng['StudentAbsenceDays'] == 'Under-7').astype(int)
        
        # Interactions
        if 'raisedhands' in df_eng.columns and 'VisITedResources' in df_eng.columns:
            df_eng['hands_resources_interaction'] = df_eng['raisedhands'] * df_eng['VisITedResources']
        
        if 'Discussion' in df_eng.columns and 'AnnouncementsView' in df_eng.columns:
            df_eng['discussion_announcements_interaction'] = df_eng['Discussion'] * df_eng['AnnouncementsView']
        
        # Categorical combinations
        if 'gender' in df_eng.columns and 'Topic' in df_eng.columns:
            df_eng['gender_topic'] = df_eng['gender'] + '_' + df_eng['Topic']
        
        if 'StageID' in df_eng.columns and 'Semester' in df_eng.columns:
            df_eng['stage_semester'] = df_eng['StageID'] + '_' + df_eng['Semester']
        
        # Binary features
        if 'engagement_score' in df_eng.columns:
            df_eng['high_engagement'] = (df_eng['engagement_score'] > df_eng['engagement_score'].median()).astype(int)
        
        if 'raisedhands' in df_eng.columns:
            df_eng['frequent_participant'] = (df_eng['raisedhands'] > df_eng['raisedhands'].median()).astype(int)
        
        return df_eng
    
    def predict(self, input_data: Union[Dict, pd.DataFrame]) -> Dict:
        """
        Make predictions on input data.
        
        Args:
            input_data: Input data as dictionary or DataFrame
            
        Returns:
            Dictionary with predictions and probabilities
        """
        if not self.loaded:
            self.load_model()
        
        # Preprocess input
        X_processed = self.preprocess_input(input_data)
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)
        
        # Convert predictions back to original labels
        predicted_labels = self.le_target.inverse_transform(predictions)
        
        # Create results
        results = {
            'predictions': predicted_labels.tolist(),
            'probabilities': probabilities.tolist(),
            'class_names': config.TARGET_LABELS,
            'model_used': self.model_name
        }
        
        return results
    
    def predict_single(self, student_data: Dict) -> Dict:
        """
        Make prediction for a single student.
        
        Args:
            student_data: Dictionary with student information
            
        Returns:
            Dictionary with prediction details
        """
        results = self.predict(student_data)
        
        prediction = results['predictions'][0]
        probabilities = results['probabilities'][0]
        
        # Create detailed result for single prediction
        detailed_result = {
            'predicted_class': prediction,
            'confidence': max(probabilities),
            'class_probabilities': {
                config.TARGET_LABELS[i]: prob for i, prob in enumerate(probabilities)
            },
            'model_used': self.model_name,
            'input_data': student_data
        }
        
        return detailed_result
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self.loaded:
            return {'error': 'Model not loaded'}
        
        info = {
            'model_name': self.model_name,
            'model_type': type(self.model).__name__,
            'feature_count': len(self.feature_names),
            'target_classes': config.TARGET_LABELS,
            'numerical_features': len(self.numerical_features),
            'categorical_features': len(self.categorical_features)
        }
        
        return info

def create_sample_input() -> Dict:
    """
    Create a sample input for testing the predictor.
    
    Returns:
        Dictionary with sample student data
    """
    sample_data = {
        'gender': 'M',
        'NationalITy': 'KW',
        'PlaceofBirth': 'KuwaIT',
        'StageID': 'lowerlevel',
        'GradeID': 'G-04',
        'SectionID': 'A',
        'Topic': 'IT',
        'Semester': 'F',
        'Relation': 'Father',
        'raisedhands': 15,
        'VisITedResources': 16,
        'AnnouncementsView': 2,
        'Discussion': 20,
        'ParentAnsweringSurvey': 'Yes',
        'ParentschoolSatisfaction': 'Good',
        'StudentAbsenceDays': 'Under-7'
    }
    
    return sample_data

# Example usage
if __name__ == "__main__":
    # Create predictor instance
    predictor = StudentPerformancePredictor()
    
    try:
        # Load model
        predictor.load_model()
        
        # Create sample input
        sample_input = create_sample_input()
        
        print("Sample Student Data:")
        for key, value in sample_input.items():
            print(f"  {key}: {value}")
        
        # Make prediction
        result = predictor.predict_single(sample_input)
        
        print("\\nPrediction Result:")
        print(f"  Predicted Class: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Class Probabilities:")
        for class_name, prob in result['class_probabilities'].items():
            print(f"    {class_name}: {prob:.4f}")
        
        # Get model info
        model_info = predictor.get_model_info()
        print(f"\\nModel Information:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure to run the main pipeline first to train the models.")
