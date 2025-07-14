"""
Advanced feature engineering module.
Provides comprehensive feature preprocessing, engineering, and selection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
from config import config

warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """Advanced feature engineering and preprocessing class."""
    
    def __init__(self, data_path: str = None):
        """Initialize the feature engineer with data."""
        self.data_path = data_path or config.DATA_PATH
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.scalers = {}
        self.encoders = {}
        self.feature_selectors = {}
        self.load_data()
        
    def load_data(self):
        """Load the dataset."""
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {self.df.shape}")
        
    def create_advanced_features(self):
        """Create advanced engineered features."""
        print("Creating advanced features...")
        
        df_engineered = self.df.copy()
        
        # Engagement Score (combination of interaction features)
        engagement_features = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']
        df_engineered['engagement_score'] = df_engineered[engagement_features].sum(axis=1)
        df_engineered['engagement_score_normalized'] = (df_engineered['engagement_score'] - 
                                                       df_engineered['engagement_score'].min()) / (
                                                       df_engineered['engagement_score'].max() - 
                                                       df_engineered['engagement_score'].min())
        
        # Participation Rate
        df_engineered['participation_rate'] = (df_engineered['raisedhands'] + 
                                              df_engineered['Discussion']) / 2
        
        # Digital Engagement
        df_engineered['digital_engagement'] = (df_engineered['VisITedResources'] + 
                                             df_engineered['AnnouncementsView']) / 2
        
        # Interaction Ratios
        df_engineered['hands_to_discussion_ratio'] = (df_engineered['raisedhands'] + 1) / (df_engineered['Discussion'] + 1)
        df_engineered['resources_to_announcements_ratio'] = (df_engineered['VisITedResources'] + 1) / (df_engineered['AnnouncementsView'] + 1)
        
        # Binary features
        df_engineered['high_engagement'] = (df_engineered['engagement_score'] > 
                                          df_engineered['engagement_score'].median()).astype(int)
        
        df_engineered['frequent_participant'] = (df_engineered['raisedhands'] > 
                                               df_engineered['raisedhands'].median()).astype(int)
        
        # Grade level features
        df_engineered['grade_numeric'] = df_engineered['GradeID'].str.extract('(\d+)').astype(float)
        df_engineered['is_high_grade'] = (df_engineered['grade_numeric'] > 6).astype(int)
        
        # Parent involvement score
        parent_involvement = 0
        parent_involvement += (df_engineered['ParentAnsweringSurvey'] == 'Yes').astype(int)
        parent_involvement += (df_engineered['ParentschoolSatisfaction'] == 'Good').astype(int)
        df_engineered['parent_involvement_score'] = parent_involvement
        
        # Attendance quality
        df_engineered['good_attendance'] = (df_engineered['StudentAbsenceDays'] == 'Under-7').astype(int)
        
        # Interaction polynomial features
        df_engineered['hands_resources_interaction'] = (df_engineered['raisedhands'] * 
                                                       df_engineered['VisITedResources'])
        df_engineered['discussion_announcements_interaction'] = (df_engineered['Discussion'] * 
                                                               df_engineered['AnnouncementsView'])
        
        # Categorical combinations
        df_engineered['gender_topic'] = df_engineered['gender'] + '_' + df_engineered['Topic']
        df_engineered['stage_semester'] = df_engineered['StageID'] + '_' + df_engineered['Semester']
        
        print(f"Features created. New shape: {df_engineered.shape}")
        self.df_engineered = df_engineered
        return df_engineered
        
    def preprocess_features(self):
        """Preprocess features for machine learning."""
        print("Preprocessing features...")
        
        if not hasattr(self, 'df_engineered'):
            self.create_advanced_features()
            
        df_processed = self.df_engineered.copy()
        
        # Separate features and target
        X = df_processed.drop(config.TARGET_COLUMN, axis=1)
        y = df_processed[config.TARGET_COLUMN]
        
        # Encode target variable
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)
        
        # Separate numerical and categorical features
        numerical_features = []
        categorical_features = []
        
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                numerical_features.append(col)
            else:
                categorical_features.append(col)
                
        print(f"Numerical features: {len(numerical_features)}")
        print(f"Categorical features: {len(categorical_features)}")
        
        # Handle numerical features
        X_numerical = X[numerical_features].copy()
        
        # Handle missing values in numerical features
        X_numerical = X_numerical.fillna(X_numerical.median())
        
        # Scale numerical features
        scaler = RobustScaler()
        X_numerical_scaled = pd.DataFrame(
            scaler.fit_transform(X_numerical),
            columns=numerical_features,
            index=X_numerical.index
        )
        
        # Handle categorical features
        X_categorical = X[categorical_features].copy()
        
        # Handle missing values in categorical features
        X_categorical = X_categorical.fillna('Unknown')
        
        # One-hot encode categorical features
        X_categorical_encoded = pd.get_dummies(X_categorical, prefix=categorical_features, drop_first=True)
        
        # Combine processed features
        X_processed = pd.concat([X_numerical_scaled, X_categorical_encoded], axis=1)
        
        print(f"Processed features shape: {X_processed.shape}")
        
        # Store preprocessing objects
        self.scaler = scaler
        self.le_target = le_target
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.feature_names = X_processed.columns.tolist()
        
        self.X = X_processed
        self.y = y_encoded
        
        return X_processed, y_encoded
        
    def select_features(self, method='mutual_info', k=15):
        """Select the most important features."""
        print(f"Selecting top {k} features using {method}...")
        
        if self.X is None or self.y is None:
            self.preprocess_features()
            
        # Define feature selection methods
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        elif method == 'chi2':
            # Chi2 requires non-negative features
            X_positive = self.X - self.X.min() + 1e-6
            selector = SelectKBest(score_func=chi2, k=k)
            X_selected = selector.fit_transform(X_positive, self.y)
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=k)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
            
        if method != 'chi2':
            X_selected = selector.fit_transform(self.X, self.y)
            
        # Get selected feature names
        selected_features = [self.feature_names[i] for i in selector.get_support(indices=True)]
        
        print(f"Selected features: {selected_features}")
        
        # Store selector
        self.feature_selectors[method] = selector
        
        return X_selected, selected_features
        
    def apply_dimensionality_reduction(self, method='pca', n_components=10):
        """Apply dimensionality reduction techniques."""
        print(f"Applying {method} with {n_components} components...")
        
        if self.X is None:
            self.preprocess_features()
            
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=config.RANDOM_STATE)
            X_reduced = reducer.fit_transform(self.X)
            
            # Print explained variance ratio
            print(f"Explained variance ratio: {reducer.explained_variance_ratio_}")
            print(f"Total explained variance: {reducer.explained_variance_ratio_.sum():.4f}")
            
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=config.RANDOM_STATE, 
                          perplexity=30, n_iter=300)
            X_reduced = reducer.fit_transform(self.X)
            
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
            
        return X_reduced, reducer
        
    def create_feature_interactions(self):
        """Create feature interactions and polynomial features."""
        print("Creating feature interactions...")
        
        if self.X is None:
            self.preprocess_features()
            
        from sklearn.preprocessing import PolynomialFeatures
        
        # Select only numerical features for polynomial features
        numerical_cols = [col for col in self.X.columns if col in self.numerical_features]
        X_numerical = self.X[numerical_cols]
        
        # Create polynomial features (degree 2)
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_poly = poly.fit_transform(X_numerical)
        
        # Create feature names for polynomial features
        poly_feature_names = poly.get_feature_names_out(numerical_cols)
        
        # Combine with original categorical features
        categorical_cols = [col for col in self.X.columns if col not in self.numerical_features]
        X_categorical = self.X[categorical_cols]
        
        # Create final feature matrix
        X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=self.X.index)
        X_combined = pd.concat([X_poly_df, X_categorical], axis=1)
        
        print(f"Features with interactions shape: {X_combined.shape}")
        
        return X_combined, poly
        
    def get_feature_statistics(self):
        """Get comprehensive feature statistics."""
        if self.X is None:
            self.preprocess_features()
            
        stats = {
            'total_features': len(self.feature_names),
            'numerical_features': len(self.numerical_features),
            'categorical_features': len(self.categorical_features),
            'feature_names': self.feature_names,
            'data_shape': self.X.shape,
            'target_distribution': pd.Series(self.y).value_counts().to_dict()
        }
        
        return stats
        
    def save_preprocessing_objects(self):
        """Save preprocessing objects for later use."""
        import joblib
        
        preprocessing_objects = {
            'scaler': self.scaler,
            'le_target': self.le_target,
            'feature_names': self.feature_names,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'feature_selectors': self.feature_selectors
        }
        
        joblib.dump(preprocessing_objects, f'{config.MODELS_DIR}/preprocessing_objects.pkl')
        print("Preprocessing objects saved!")
        
    def run_complete_feature_engineering(self):
        """Run the complete feature engineering pipeline."""
        print("Starting Advanced Feature Engineering...")
        
        # Create advanced features
        df_engineered = self.create_advanced_features()
        
        # Preprocess features
        X_processed, y_encoded = self.preprocess_features()
        
        # Feature selection
        X_selected, selected_features = self.select_features(method='mutual_info', k=15)
        
        # Get statistics
        stats = self.get_feature_statistics()
        
        # Save preprocessing objects
        self.save_preprocessing_objects()
        
        print("\nFeature Engineering Summary:")
        print(f"Original features: {self.df.shape[1] - 1}")
        print(f"Engineered features: {len(stats['feature_names'])}")
        print(f"Selected features: {len(selected_features)}")
        print(f"Final data shape: {X_processed.shape}")
        
        return X_processed, y_encoded, selected_features, stats

if __name__ == "__main__":
    engineer = AdvancedFeatureEngineer()
    X, y, selected_features, stats = engineer.run_complete_feature_engineering()
    print("Feature engineering complete!")
